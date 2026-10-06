"""Client for the CASCADE web server (Paton group, Colorado State University).

CASCADE predicts 1H and 13C chemical shifts with a graph neural network on an
MMFF conformer ensemble that the server generates itself. This module sends
the structure as a SMILES string, waits for the job, downloads the result and
maps it back onto the atoms of the host molecule.

    https://nova.chem.colostate.edu/v2/cascade/

The server has no published API. The calls below are the ones its own web
page makes:

    POST /v2/cascade/predict_NMR_C/   smiles=<SMILES>&type_=C  -> {"task_id": ...}
    POST /v2/cascade/predict_NMR_H/   smiles=<SMILES>&type_=H  -> {"task_id": ...}
    GET  /v2/cascade/check_task/?task_id=...   -> "running" | "Error1" | <html>
    GET  /v2/cascade/download/<task_id>/       -> .tar.gz with
         weighted_shift.csv   (mol_id, atom_index [1-based], Shift, Confidence)
         conformers.sdf       (the server's molecule, explicit H, one per conformer)
         conformers_shift.csv (per-conformer shifts, relative_E, b_weight)

The structure leaves the user's computer here, and only here. Nothing in
this module touches Qt, so it is unit-tested with a fake ``opener``.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_SERVER = "https://nova.chem.colostate.edu"
API_ROOT = "/v2/cascade"

#: Seconds between two status polls, and the overall limit for one job.
POLL_INTERVAL_SEC = 2.0
JOB_TIMEOUT_SEC = 600.0
#: Per-request network timeout.
HTTP_TIMEOUT_SEC = 60.0

USER_AGENT = "MoleditPy-CASCADE-plugin"

NUCLEUS_TYPE = {"1H": "H", "13C": "C"}


class CascadeError(RuntimeError):
    """A failure with a message meant for the user."""


class Cancelled(CascadeError):
    """The user cancelled while the job was running."""


# ---------------------------------------------------------------------------
# Molecule <-> SMILES
# ---------------------------------------------------------------------------


def _has_3d(mol) -> bool:
    try:
        return mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    except Exception:
        return False


def prepare_molecule(mol):
    """Sanitised copy of ``mol`` with explicit H and its stereo perceived.

    Heavy atoms keep their indices (and hydrogens already present keep
    theirs), so every index in the result is valid on the host molecule
    as long as the host already carries its hydrogens.
    """
    from rdkit import Chem

    work = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
    except Exception as exc:
        raise CascadeError(
            f"The structure could not be sanitized:\n{exc}\n\n"
            "Please correct the valences and try again."
        ) from exc
    if _has_3d(work):
        # The SMILES must carry the stereo the user built in 3D.
        Chem.AssignStereochemistryFrom3D(work)
    return Chem.AddHs(work, addCoords=_has_3d(work))


def to_smiles(mol_h) -> str:
    """Isomeric SMILES of the heavy-atom skeleton (what the server expects)."""
    from rdkit import Chem

    heavy = Chem.RemoveHs(mol_h)
    smiles = Chem.MolToSmiles(heavy, isomericSmiles=True)
    if not smiles:
        raise CascadeError("Could not write a SMILES string for this structure.")
    if "." in smiles:
        raise CascadeError(
            "The structure has more than one fragment. CASCADE predicts one molecule "
            "at a time; delete the counter-ions or solvent first."
        )
    return smiles


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _default_opener(request, timeout):
    return urllib.request.urlopen(request, timeout=timeout)


class CascadeClient:
    """One prediction job against the CASCADE server.

    ``opener(request, timeout)`` returns a file-like response (urllib's
    ``urlopen`` by default); ``sleep`` and ``clock`` are injectable so the
    polling loop can be tested without waiting.
    """

    def __init__(
        self,
        server=DEFAULT_SERVER,
        opener=None,
        sleep=time.sleep,
        clock=time.monotonic,
        poll_interval=POLL_INTERVAL_SEC,
        job_timeout=JOB_TIMEOUT_SEC,
    ):
        self.server = (server or DEFAULT_SERVER).rstrip("/")
        self.opener = opener or _default_opener
        self.sleep = sleep
        self.clock = clock
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout

    def _url(self, path: str) -> str:
        return f"{self.server}{API_ROOT}{path}"

    def _request(self, url, data=None) -> bytes:
        headers = {"User-Agent": USER_AGENT, "Referer": self._url("/home/")}
        body = None
        if data is not None:
            body = urllib.parse.urlencode(data).encode("ascii")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            with self.opener(request, HTTP_TIMEOUT_SEC) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise CascadeError(f"The CASCADE server answered HTTP {exc.code} for\n{url}") from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise CascadeError(
                f"Could not reach the CASCADE server ({self.server}):\n{reason}\n\n"
                "Check the internet connection."
            ) from exc

    # -- job steps ---------------------------------------------------------

    def submit(self, smiles: str, nucleus: str) -> str:
        """Queue a prediction and return its task id."""
        kind = NUCLEUS_TYPE.get(nucleus)
        if kind is None:
            raise CascadeError(f"Unsupported nucleus: {nucleus}")
        raw = self._request(self._url(f"/predict_NMR_{kind}/"), {"smiles": smiles, "type_": kind})
        try:
            reply = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise CascadeError("The CASCADE server sent an unexpected reply to the submission.") from exc
        task_id = reply.get("task_id") if isinstance(reply, dict) else None
        if not task_id:
            raise CascadeError(
                "The CASCADE server did not accept this molecule.\n"
                f"{(reply or {}).get('message', '') if isinstance(reply, dict) else ''}".strip()
            )
        return str(task_id)

    def wait(self, task_id: str, is_cancelled=None) -> None:
        """Poll until the job has finished; raise on failure or timeout."""
        deadline = self.clock() + self.job_timeout
        query = urllib.parse.urlencode({"task_id": task_id})
        while True:
            if is_cancelled is not None and is_cancelled():
                raise Cancelled("Cancelled.")
            text = self._request(self._url(f"/check_task/?{query}")).decode("utf-8", "replace")
            status = text.strip().strip('"')
            if status == "running":
                if self.clock() > deadline:
                    raise CascadeError(
                        f"CASCADE did not finish within {int(self.job_timeout)} s. "
                        "The server may be busy; try again later."
                    )
                self.sleep(self.poll_interval)
                continue
            if status.startswith("Error"):
                raise CascadeError(
                    "CASCADE could not find conformers for this molecule "
                    f"(server status: {status})."
                )
            return

    def download(self, task_id: str) -> dict:
        """Fetch and unpack the result archive: ``{file name: text}``."""
        raw = self._request(self._url(f"/download/{urllib.parse.quote(task_id)}/"))
        try:
            files = {}
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    handle = archive.extractfile(member)
                    if handle is not None:
                        name = member.name.rsplit("/", 1)[-1]
                        files[name] = handle.read().decode("utf-8", "replace")
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise CascadeError("The CASCADE result archive could not be read.") from exc
        if "weighted_shift.csv" not in files or "conformers.sdf" not in files:
            raise CascadeError(
                "The CASCADE result archive is missing weighted_shift.csv or conformers.sdf."
            )
        return files

    def run(self, smiles: str, nucleus: str, is_cancelled=None, progress=None) -> dict:
        """submit + wait + download."""
        if progress:
            progress("Submitting to the CASCADE server...")
        task_id = self.submit(smiles, nucleus)
        if progress:
            progress("CASCADE is searching conformers and predicting shifts...")
        self.wait(task_id, is_cancelled)
        if progress:
            progress("Downloading the result...")
        files = self.download(task_id)
        files["task_id"] = task_id
        return files


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_weighted_shifts(text: str) -> list:
    """Rows of weighted_shift.csv as ``[(atom_index_0based, ppm, confidence)]``."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            idx = int(row["atom_index"]) - 1
            ppm = float(row["Shift"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((idx, ppm, (row.get("Confidence") or "").strip()))
    if not rows:
        raise CascadeError("The CASCADE result contains no shifts.")
    return rows


def parse_conformer_count(text: str) -> int:
    """Number of distinct conformers in conformers_shift.csv (0 if absent)."""
    if not text:
        return 0
    ids = set()
    for row in csv.DictReader(io.StringIO(text)):
        ids.add(row.get("cf_id"))
    return len(ids)


def server_molecule(sdf_text: str):
    """First conformer of conformers.sdf, explicit hydrogens kept."""
    from rdkit import Chem

    block = sdf_text.split("$$$$", 1)[0]
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
    if mol is None:
        raise CascadeError("The molecule returned by CASCADE could not be read.")
    return mol


def map_server_atoms(server_mol, mol_h) -> dict:
    """Map server atom indices onto ``mol_h`` (both with explicit H).

    The server re-parses the SMILES, so its atom order need not match ours;
    a full substructure match ties the two together. Stereo is ignored for
    the match (it is the same molecule) so that a server that dropped a
    stereo flag still maps.
    """
    if server_mol.GetNumAtoms() != mol_h.GetNumAtoms():
        raise CascadeError(
            "The molecule CASCADE returned does not match the structure "
            f"({server_mol.GetNumAtoms()} vs {mol_h.GetNumAtoms()} atoms)."
        )
    match = mol_h.GetSubstructMatch(server_mol, useChirality=False)
    if not match:
        raise CascadeError("The molecule CASCADE returned could not be matched to the structure.")
    return {server_idx: host_idx for server_idx, host_idx in enumerate(match)}


def symmetrize(values: dict, classes) -> dict:
    """Average ``{atom_idx: ppm}`` over symmetry-equivalent atoms.

    Methyl protons come back from a single conformer as e.g. 1.36/1.14/1.36;
    they rotate fast and are one signal.
    """
    groups = {}
    for idx in values:
        groups.setdefault(classes[idx], []).append(idx)
    out = {}
    for members in groups.values():
        mean = sum(values[i] for i in members) / len(members)
        for i in members:
            out[i] = mean
    return out


def build_result(files: dict, mol_h, nucleus: str, symmetrize_shifts: bool = True) -> dict:
    """Turn a downloaded archive into the dialog's result dict.

    Every ``data`` item has ``idx`` (index in ``mol_h``), ``parent_idx`` (the
    heavy atom, for highlighting a hydrogen the host keeps implicit),
    ``atom``, ``ppm``, ``confidence`` and the coupling fields ``mult``,
    ``pattern``, ``j_text`` (plus ``dept``/``j_ch`` for 13C).
    """
    from . import coupling

    rows = parse_weighted_shifts(files["weighted_shift.csv"])
    mapping = map_server_atoms(server_molecule(files["conformers.sdf"]), mol_h)

    shifts, confidence = {}, {}
    for server_idx, ppm, conf in rows:
        host_idx = mapping.get(server_idx)
        if host_idx is None:
            continue
        shifts[host_idx] = ppm
        confidence[host_idx] = conf

    classes = coupling.symmetry_classes(mol_h)
    if symmetrize_shifts:
        shifts = symmetrize(shifts, classes)

    if nucleus == "1H":
        geom = coupling.ensure_3d(mol_h)
        multiplets = coupling.hh_multiplets(geom, coupling.predict_hh_couplings(geom))
    else:
        multiplets = coupling.predict_ch_couplings(mol_h)

    data = []
    for idx in sorted(shifts):
        atom = mol_h.GetAtomWithIdx(idx)
        nbrs = atom.GetNeighbors()
        parent = nbrs[0].GetIdx() if atom.GetAtomicNum() == 1 and nbrs else idx
        info = multiplets.get(idx, {})
        item = {
            "idx": idx,
            "parent_idx": parent,
            "atom": atom.GetSymbol(),
            "ppm": shifts[idx],
            "confidence": confidence.get(idx, ""),
            "mult": info.get("mult", ""),
            "pattern": info.get("pattern", []),
            "j_text": info.get("j_text", ""),
        }
        if nucleus == "13C":
            item["dept"] = info.get("dept", "")
            item["j_ch"] = info.get("j_ch")
        data.append(item)

    return {
        "nucleus": nucleus,
        "data": data,
        "mol_with_h": mol_h,
        "n_conformers": parse_conformer_count(files.get("conformers_shift.csv", "")),
        "task_id": files.get("task_id", ""),
    }


def predict(mol, nucleus: str, server=DEFAULT_SERVER, opener=None, is_cancelled=None,
            progress=None, symmetrize_shifts=True, client=None) -> dict:
    """Whole round trip for one molecule; blocking — run it off the UI thread."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise CascadeError("Please draw or load a molecule first.")
    mol_h = prepare_molecule(mol)
    target = 6 if nucleus == "13C" else 1
    if not any(a.GetAtomicNum() == target for a in mol_h.GetAtoms()):
        raise CascadeError(f"The structure has no {'carbon' if target == 6 else 'hydrogen'} atoms.")
    smiles = to_smiles(mol_h)
    client = client or CascadeClient(server=server, opener=opener)
    files = client.run(smiles, nucleus, is_cancelled=is_cancelled, progress=progress)
    result = build_result(files, mol_h, nucleus, symmetrize_shifts=symmetrize_shifts)
    result["smiles"] = smiles
    return result
