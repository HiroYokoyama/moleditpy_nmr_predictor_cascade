"""Everything the CASCADE predictor does that does not need Qt.

Conformer generation and Boltzmann weighting run in MoleditPy's own Python
(RDKit is the host's); the neural network runs in the separate CASCADE-2.0
environment through ``cascade_bridge.py``. This module builds the request,
runs the bridge, and folds the per-conformer answers back into one shift per
carbon.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_SCRIPT = os.path.join(PLUGIN_DIR, "cascade_bridge.py")

#: Elements the CASCADE-2.0 tokenizer was trained on (atomic numbers).
#: Anything else is mapped to an "unknown" token and predicted badly.
SUPPORTED_ELEMENTS = {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}

#: Model folders inside a CASCADE-2.0 checkout, best first.
MODEL_SUBDIRS = (
    os.path.join("models", "Predict_SMILES_FF_GPR"),
    os.path.join("models", "Predict_SMILES_FF"),
)

#: The GPR model's 95% half-width (ppm) above which the notebook calls a
#: prediction "Moderate" or "Low" confidence.
CONFIDENCE_LOW = 1.485774
CONFIDENCE_MODERATE = 1.419831

#: kcal/mol per K
GAS_CONSTANT_KCAL = 0.0019872041

#: Upper bound for one bridge run. TensorFlow alone takes 10-30 s to start.
BRIDGE_TIMEOUT_SEC = 900


class CascadeError(RuntimeError):
    """A failure with a message meant for the user."""


def default_settings() -> dict:
    return {
        "python_path": "",
        "model_dir": "",
        "geometry": "ensemble",  # or "current"
        "num_conformers": 30,
        "energy_window": 3.0,  # kcal/mol
        "rms_prune": 0.5,  # Angstrom
        "temperature": 298.15,  # K
        "symmetrize": True,
        "random_seed": 42,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def model_kind(model_dir: str):
    """'gpr', 'plain' or None — mirrors ``cascade_bridge.model_kind``."""
    if not model_dir or not os.path.isfile(os.path.join(model_dir, "preprocessor_orig.p")):
        return None
    if os.path.isfile(os.path.join(model_dir, "best_model_val_mae.h5")) and os.path.isfile(
        os.path.join(model_dir, "model.py")
    ):
        return "gpr"
    if os.path.isfile(os.path.join(model_dir, "best_model.h5")):
        return "plain"
    return None


def resolve_model_dir(path: str):
    """Accept a model folder or a whole CASCADE-2.0 checkout.

    Returns the model folder to use, or None when ``path`` holds neither.
    """
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if model_kind(path):
        return path
    for sub in MODEL_SUBDIRS:
        candidate = os.path.join(path, sub)
        if model_kind(candidate):
            return candidate
    return None


def validate_settings(settings: dict) -> tuple:
    """Return ``(python_path, model_dir)`` or raise ``CascadeError``."""
    python_path = (settings.get("python_path") or "").strip()
    if not python_path:
        raise CascadeError(
            "The CASCADE Python interpreter is not set.\n"
            "Open Analysis > NMR Prediction (CASCADE) > Settings and point it at the "
            "python executable of your CASCADE-2.0 environment."
        )
    resolved = shutil.which(python_path) if not os.path.isfile(python_path) else python_path
    if not resolved or not os.path.isfile(resolved):
        raise CascadeError(f"The CASCADE Python interpreter was not found:\n{python_path}")

    model_dir = resolve_model_dir(settings.get("model_dir") or "")
    if model_dir is None:
        raise CascadeError(
            "No CASCADE-2.0 model was found in the configured folder:\n"
            f"{settings.get('model_dir') or '(not set)'}\n\n"
            "Choose your CASCADE-2.0 checkout, or its models/Predict_SMILES_FF_GPR folder."
        )
    return resolved, model_dir


def unsupported_elements(mol) -> list:
    """Symbols of elements the model was not trained on."""
    return sorted(
        {a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in SUPPORTED_ELEMENTS}
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _has_3d(mol) -> bool:
    try:
        return mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    except Exception:
        return False


def prepare_molecule(mol):
    """Sanitised copy with explicit hydrogens and the stereo of ``mol``.

    Heavy atoms keep their indices (hydrogens already present keep theirs
    too), so carbon indices map straight back onto the host molecule.
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
        # Read the stereo the user built in 3D before re-embedding drops it.
        Chem.AssignStereochemistryFrom3D(work)
    return Chem.AddHs(work, addCoords=_has_3d(work))


def boltzmann_weights(energies, temperature: float = 298.15) -> list:
    """Normalised Boltzmann weights for energies in kcal/mol."""
    if not energies:
        return []
    e_min = min(energies)
    kt = GAS_CONSTANT_KCAL * float(temperature)
    raw = [math.exp(-(e - e_min) / kt) for e in energies]
    total = sum(raw)
    return [r / total for r in raw]


def _ff_energies(mol, conf_ids, max_iters=2000):
    """Optimise every conformer; MMFF94 when parameterised, else UFF."""
    from rdkit.Chem import AllChem

    if AllChem.MMFFHasAllMoleculeParams(mol):
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        field = "MMFF94"
    else:
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        field = "UFF"
    energies = {}
    for cid, (_converged, energy) in zip(conf_ids, results):
        energies[cid] = energy
    return energies, field


def generate_conformers(mol_h, settings: dict) -> dict:
    """Conformers to predict on, with their Boltzmann weights.

    Returns ``{"mols": [Mol, ...], "weights": [...], "energies": [...],
    "force_field": str}``; every Mol holds one conformer of ``mol_h``.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if settings.get("geometry") == "current":
        if not _has_3d(mol_h):
            raise CascadeError(
                "'Current geometry' needs 3D coordinates. Convert the structure to 3D "
                "first, or switch the geometry option to 'Conformer ensemble'."
            )
        single = Chem.Mol(mol_h, confId=mol_h.GetConformer().GetId())
        return {"mols": [single], "weights": [1.0], "energies": [0.0], "force_field": "input"}

    work = Chem.Mol(mol_h)
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(settings.get("random_seed", 42))
    params.pruneRmsThresh = float(settings.get("rms_prune", 0.5))
    n_conf = max(1, int(settings.get("num_conformers", 30)))
    conf_ids = list(AllChem.EmbedMultipleConfs(work, numConfs=n_conf, params=params))
    if not conf_ids:
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(work, numConfs=n_conf, params=params))
    if not conf_ids:
        raise CascadeError("RDKit could not generate any 3D conformer for this structure.")

    energies, field = _ff_energies(work, conf_ids)
    e_min = min(energies.values())
    window = float(settings.get("energy_window", 3.0))
    kept = sorted(
        (cid for cid in conf_ids if energies[cid] - e_min <= window),
        key=lambda cid: energies[cid],
    )
    rel = [energies[cid] - e_min for cid in kept]
    weights = boltzmann_weights(rel, settings.get("temperature", 298.15))
    mols = [Chem.Mol(work, confId=cid) for cid in kept]
    return {"mols": mols, "weights": weights, "energies": rel, "force_field": field}


def carbon_indices(mol) -> list:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]


def build_request(mols, atom_index) -> dict:
    from rdkit import Chem

    return {
        "molecules": [
            {"molblock": Chem.MolToMolBlock(m), "atom_index": list(atom_index)} for m in mols
        ]
    }


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def _startupinfo():
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return info


def run_bridge(
    python_path: str,
    model_dir: str,
    request=None,
    check: bool = False,
    timeout: float = BRIDGE_TIMEOUT_SEC,
    runner=subprocess.run,
) -> dict:
    """Run ``cascade_bridge.py`` in the CASCADE environment and return its JSON.

    ``runner`` is injectable for tests. Raises ``CascadeError`` on any
    failure, with the bridge's own message when it produced one.
    """
    tmpdir = tempfile.mkdtemp(prefix="cascade_nmr_")
    try:
        out_path = os.path.join(tmpdir, "out.json")
        cmd = [python_path, BRIDGE_SCRIPT, "--model-dir", model_dir, "--output", out_path]
        if check:
            cmd.append("--check")
        else:
            in_path = os.path.join(tmpdir, "in.json")
            with open(in_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(request, handle)
            cmd += ["--input", in_path]

        env = dict(os.environ)
        # Keep MoleditPy's own packages out of the CASCADE interpreter.
        for var in ("PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH"):
            env.pop(var, None)
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = runner(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                cwd=tmpdir,
                startupinfo=_startupinfo(),
            )
        except subprocess.TimeoutExpired as exc:
            raise CascadeError(
                f"CASCADE did not finish within {int(timeout)} s.\n"
                "Try fewer conformers or a smaller molecule."
            ) from exc
        except OSError as exc:
            raise CascadeError(f"Could not start the CASCADE Python:\n{exc}") from exc

        payload = None
        if os.path.isfile(out_path):
            try:
                with open(out_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                payload = None
        if payload is None:
            tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
            raise CascadeError(
                f"CASCADE exited with code {proc.returncode} without a result.\n\n{tail}"
            )
        if not payload.get("ok"):
            raise CascadeError(f"CASCADE reported an error:\n{payload.get('error', 'unknown')}")
        return payload
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def confidence_label(std_ppm):
    """'High' / 'Moderate' / 'Low' from a GPR standard deviation (ppm)."""
    if std_ppm is None:
        return ""
    half_width = 1.96 * std_ppm
    if half_width >= CONFIDENCE_LOW:
        return "Low"
    if half_width >= CONFIDENCE_MODERATE:
        return "Moderate"
    return "High"


def average_conformers(results, weights, atom_index) -> dict:
    """Boltzmann-average the bridge results.

    Returns ``{atom_idx: {"ppm", "std", "spread"}}``: ``std`` is the weighted
    model uncertainty (None for the plain model) and ``spread`` the weighted
    standard deviation of the shift over the conformers.
    """
    if len(results) != len(weights):
        raise CascadeError(
            f"CASCADE returned {len(results)} conformers, {len(weights)} were sent."
        )
    out = {}
    for pos, idx in enumerate(atom_index):
        values = [r["shift"][pos] for r in results]
        mean = sum(w * v for w, v in zip(weights, values))
        spread = math.sqrt(max(0.0, sum(w * (v - mean) ** 2 for w, v in zip(weights, values))))
        std = None
        if all(r.get("std") for r in results):
            std = sum(w * r["std"][pos] for w, r in zip(weights, results))
        out[idx] = {"ppm": mean, "std": std, "spread": spread}
    return out


def symmetrize(per_atom: dict, classes) -> dict:
    """Average shifts over topologically equivalent carbons.

    Conformer sampling breaks the symmetry a little (the two ortho carbons of
    a phenyl ring come back 0.1 ppm apart); equivalent nuclei should show as
    one line.
    """
    groups = {}
    for idx in per_atom:
        groups.setdefault(classes[idx], []).append(idx)
    out = {}
    for members in groups.values():
        n = len(members)
        ppm = sum(per_atom[i]["ppm"] for i in members) / n
        stds = [per_atom[i]["std"] for i in members]
        std = None if any(s is None for s in stds) else sum(stds) / n
        spread = max(per_atom[i]["spread"] for i in members)
        for i in members:
            out[i] = {"ppm": ppm, "std": std, "spread": spread}
    return out


def predict(mol, settings: dict, runner=subprocess.run, progress=None) -> dict:
    """Full prediction for one molecule; blocking (run it off the UI thread).

    Returns ``{"nucleus": "13C", "data": [...], "mol_with_h": Mol,
    "model": kind, "n_conformers": int, "force_field": str}`` where every
    ``data`` item is a dict with ``idx, atom, ppm, std, confidence, spread``
    plus the C-H coupling fields from ``coupling.predict_ch_couplings``.
    """
    from . import coupling

    def report(text):
        if progress is not None:
            progress(text)

    python_path, model_dir = validate_settings(settings)
    if mol is None or mol.GetNumAtoms() == 0:
        raise CascadeError("Please draw or load a molecule first.")

    bad = unsupported_elements(mol)
    if bad:
        raise CascadeError(
            "CASCADE-2.0 was trained on H, C, N, O, F, Si, P, S, Cl, Br and I only.\n"
            f"This structure contains: {', '.join(bad)}"
        )

    mol_h = prepare_molecule(mol)
    c_idx = carbon_indices(mol_h)
    if not c_idx:
        raise CascadeError("The structure has no carbon atoms.")

    report("Generating conformers...")
    ensemble = generate_conformers(mol_h, settings)

    report(f"Running CASCADE on {len(ensemble['mols'])} conformer(s)...")
    payload = run_bridge(python_path, model_dir, build_request(ensemble["mols"], c_idx), runner=runner)

    per_atom = average_conformers(payload.get("results", []), ensemble["weights"], c_idx)
    if settings.get("symmetrize", True):
        per_atom = symmetrize(per_atom, coupling.symmetry_classes(mol_h))

    ch = coupling.predict_ch_couplings(mol_h)
    data = []
    for idx in c_idx:
        item = per_atom[idx]
        info = ch.get(idx, {})
        data.append(
            {
                "idx": idx,
                "parent_idx": idx,
                "atom": "C",
                "ppm": item["ppm"],
                "std": item["std"],
                "spread": item["spread"],
                "confidence": confidence_label(item["std"]),
                "mult": info.get("mult", ""),
                "dept": info.get("dept", ""),
                "j_ch": info.get("j_ch"),
                "pattern": info.get("pattern", []),
                "j_text": info.get("j_text", ""),
            }
        )

    # The display copy keeps the lowest-energy conformer for 3D labels.
    display = ensemble["mols"][0] if ensemble["mols"] else mol_h
    return {
        "nucleus": "13C",
        "data": data,
        "mol_with_h": display,
        "model": payload.get("model", ""),
        "n_conformers": len(ensemble["mols"]),
        "force_field": ensemble["force_field"],
        "unknown_elements": payload.get("unknown_elements", []),
    }


def python_executable_hint() -> str:
    """Where a conda env usually keeps python, for the settings dialog."""
    if sys.platform.startswith("win"):
        return r"e.g. C:\Users\you\miniconda3\envs\cascadeV2\python.exe"
    return "e.g. ~/miniconda3/envs/cascadeV2/bin/python"
