"""cascade_client tests against a fake CASCADE server (no network).

The fake answers the three endpoints the real server has; result archives
are built here from RDKit molecules in the server's format (see the module
docstring of cascade_client).
"""

import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.parse

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmr_predictor_cascade import cascade_client as cc  # noqa: E402


# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------


def make_archive(server_mol, shifts, confidence="High", n_conformers=1):
    """A result .tar.gz: ``shifts`` maps 0-based server atom index -> ppm."""
    weighted = io.StringIO()
    weighted.write(",mol_id,atom_index,Shift,Confidence\n")
    for row, (idx, ppm) in enumerate(sorted(shifts.items())):
        weighted.write(f"{row},0,{idx + 1},{ppm},{confidence}\n")

    per_conf = io.StringIO()
    per_conf.write(",atom_index,cf_id,mol_id,relative_E,predicted,b_weight\n")
    row = 0
    for cf in range(n_conformers):
        for idx, ppm in sorted(shifts.items()):
            per_conf.write(f"{row},{idx + 1},{cf},0,0.0,{ppm},{1.0 / n_conformers}\n")
            row += 1

    sdf = Chem.MolToMolBlock(server_mol) + "$$$$\n"
    files = {
        "weighted_shift.csv": weighted.getvalue(),
        "conformers_shift.csv": per_conf.getvalue(),
        "conformers.sdf": sdf,
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeServer:
    """Callable with urlopen's signature; records every request."""

    def __init__(self, archive=b"", polls_running=0, status="<html>done</html>", submit_reply=None):
        self.archive = archive
        self.polls_left = polls_running
        self.status = status
        self.submit_reply = submit_reply or {"message": "queued", "task_id": "task42"}
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        url = request.full_url
        if "/predict_NMR_" in url:
            return _Response(json.dumps(self.submit_reply).encode())
        if "/check_task/" in url:
            if self.polls_left > 0:
                self.polls_left -= 1
                return _Response(b'"running"')
            return _Response(self.status.encode())
        if "/download/" in url:
            return _Response(self.archive)
        raise AssertionError(f"unexpected URL {url}")


def _client(server, **kwargs):
    return cc.CascadeClient(opener=server, sleep=lambda s: None, **kwargs)


def _embedded(smiles, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


# ---------------------------------------------------------------------------
# HTTP steps
# ---------------------------------------------------------------------------


def test_submit_posts_smiles_and_type():
    server = FakeServer()
    assert _client(server).submit("CCO", "13C") == "task42"
    request = server.requests[0]
    assert request.full_url == "https://nova.chem.colostate.edu/v2/cascade/predict_NMR_C/"
    assert request.get_method() == "POST"
    assert urllib.parse.parse_qs(request.data.decode()) == {"smiles": ["CCO"], "type_": ["C"]}


def test_submit_1h_uses_the_h_queue():
    server = FakeServer()
    _client(server).submit("CCO", "1H")
    assert server.requests[0].full_url.endswith("/predict_NMR_H/")


def test_submit_rejected_molecule():
    server = FakeServer(submit_reply={"message": "not allowed", "task_id": None})
    with pytest.raises(cc.CascadeError, match="did not accept"):
        _client(server).submit("[U]", "13C")


def test_submit_unknown_nucleus():
    with pytest.raises(cc.CascadeError, match="Unsupported nucleus"):
        _client(FakeServer()).submit("CCO", "19F")


def test_custom_server_url():
    server = FakeServer()
    cc.CascadeClient(server="http://localhost:8000/", opener=server).submit("C", "13C")
    assert server.requests[0].full_url == "http://localhost:8000/v2/cascade/predict_NMR_C/"


def test_wait_polls_until_done():
    server = FakeServer(polls_running=3)
    sleeps = []
    client = cc.CascadeClient(opener=server, sleep=sleeps.append, poll_interval=0.5)
    client.wait("task42")
    assert len(sleeps) == 3 and sleeps[0] == 0.5
    assert "task_id=task42" in server.requests[-1].full_url


def test_wait_reports_failed_job():
    with pytest.raises(cc.CascadeError, match="could not find conformers"):
        _client(FakeServer(status='"Error1"')).wait("task42")


def test_wait_times_out():
    ticks = iter(range(0, 10000, 100))
    client = cc.CascadeClient(
        opener=FakeServer(polls_running=10**6),
        sleep=lambda s: None,
        clock=lambda: next(ticks),
        job_timeout=250,
    )
    with pytest.raises(cc.CascadeError, match="did not finish"):
        client.wait("task42")


def test_wait_can_be_cancelled():
    with pytest.raises(cc.Cancelled):
        _client(FakeServer(polls_running=5)).wait("task42", is_cancelled=lambda: True)


def test_network_errors_become_cascade_errors():
    def offline(request, timeout):
        raise urllib.error.URLError("no route to host")

    with pytest.raises(cc.CascadeError, match="Could not reach"):
        _client(offline).submit("C", "13C")

    def http_500(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)

    with pytest.raises(cc.CascadeError, match="HTTP 500"):
        _client(http_500).submit("C", "13C")


def test_download_rejects_garbage():
    with pytest.raises(cc.CascadeError, match="could not be read"):
        _client(FakeServer(archive=b"not a tarball")).download("task42")


def test_download_requires_both_files():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("weighted_shift.csv")
        info.size = 0
        archive.addfile(info, io.BytesIO(b""))
    with pytest.raises(cc.CascadeError, match="missing"):
        _client(FakeServer(archive=buf.getvalue())).download("task42")


# ---------------------------------------------------------------------------
# Parsing and mapping
# ---------------------------------------------------------------------------


def test_parse_weighted_shifts():
    text = ",mol_id,atom_index,Shift,Confidence\n0,0,1,18.22,Moderate\n1,0,2,58.14,High\n"
    assert cc.parse_weighted_shifts(text) == [(0, 18.22, "Moderate"), (1, 58.14, "High")]
    with pytest.raises(cc.CascadeError):
        cc.parse_weighted_shifts(",mol_id,atom_index,Shift,Confidence\n")


def test_map_server_atoms_handles_a_different_atom_order():
    server_mol = _embedded("CCO")  # server: C0 C1 O2
    host = Chem.AddHs(Chem.MolFromSmiles("OCC"))  # host: O0 C1 C2
    mapping = cc.map_server_atoms(server_mol, host)
    assert host.GetAtomWithIdx(mapping[0]).GetDegree() == 4  # CH3 carbon
    assert mapping[2] == 0  # oxygen


def test_map_server_atoms_rejects_another_molecule():
    with pytest.raises(cc.CascadeError, match="does not match"):
        cc.map_server_atoms(_embedded("CCC"), Chem.AddHs(Chem.MolFromSmiles("CCO")))


def test_symmetrize_averages_methyl_protons():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    classes = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    values = {2: 1.36, 3: 1.14, 4: 1.36}
    out = cc.symmetrize(values, classes)
    assert out[2] == out[3] == out[4] == pytest.approx((1.36 + 1.14 + 1.36) / 3)


# ---------------------------------------------------------------------------
# Whole round trip
# ---------------------------------------------------------------------------


def test_predict_13c_round_trip():
    server_mol = _embedded("CCO")
    archive = make_archive(server_mol, {0: 18.22, 1: 58.14}, n_conformers=3)
    host = Chem.AddHs(Chem.MolFromSmiles("OCC"))
    result = cc.predict(host, "13C", opener=FakeServer(archive=archive))

    assert result["nucleus"] == "13C"
    assert result["n_conformers"] == 3
    assert result["task_id"] == "task42"
    by_idx = {item["idx"]: item for item in result["data"]}
    assert by_idx[1]["ppm"] == 58.14 and by_idx[1]["dept"] == "CH2" and by_idx[1]["mult"] == "t"
    assert by_idx[2]["ppm"] == 18.22 and by_idx[2]["j_ch"] == 125.0
    assert all(item["confidence"] == "High" for item in result["data"])


def test_predict_1h_round_trip_with_couplings():
    server_mol = _embedded("CCO")
    # server H order: 3,4,5 on C0; 6,7 on C1; 8 on O (not reported)
    shifts = {3: 1.36, 4: 1.14, 5: 1.36, 6: 3.76, 7: 3.76}
    archive = make_archive(server_mol, shifts)
    host = _embedded("CCO", seed=11)
    result = cc.predict(host, "1H", opener=FakeServer(archive=archive))

    data = result["data"]
    assert len(data) == 5
    methyl = [d for d in data if d["parent_idx"] == 0]
    assert len(methyl) == 3
    assert all(d["ppm"] == pytest.approx(1.2867, abs=1e-3) for d in methyl)  # averaged
    assert all(d["mult"] == "t" and d["j_text"] == "7.0" for d in methyl)
    ch2 = [d for d in data if d["parent_idx"] == 1]
    assert all(d["mult"] == "q" for d in ch2)


def test_predict_without_symmetrizing_keeps_raw_values():
    server_mol = _embedded("CC")
    archive = make_archive(server_mol, {2: 1.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0})
    result = cc.predict(
        Chem.AddHs(Chem.MolFromSmiles("CC")), "1H",
        opener=FakeServer(archive=archive), symmetrize_shifts=False,
    )
    assert sorted(d["ppm"] for d in result["data"]) == [1.0] * 5 + [2.0]


def test_predict_sends_stereo_read_from_3d():
    """A 3D alkene without stereo flags must still be sent as E."""
    server = FakeServer(archive=make_archive(_embedded("C/C=C/C"), {0: 17.0}))
    host = Chem.AddHs(Chem.MolFromSmiles("C/C=C/C"))
    AllChem.EmbedMolecule(host, randomSeed=5)
    Chem.RemoveStereochemistry(host)
    cc.predict(host, "13C", opener=server)
    sent = urllib.parse.parse_qs(server.requests[0].data.decode())["smiles"][0]
    assert sent in ("C/C=C/C", "C\\C=C\\C")


def test_predict_reports_progress():
    messages = []
    archive = make_archive(_embedded("C"), {0: -2.0})
    cc.predict(Chem.MolFromSmiles("C"), "13C", opener=FakeServer(archive=archive), progress=messages.append)
    assert messages[0].startswith("Submitting") and messages[-1].startswith("Downloading")


@pytest.mark.parametrize(
    "smiles, nucleus, message",
    [
        ("CCO.Cl", "13C", "more than one fragment"),
        ("O", "13C", "no carbon"),
        ("ClC(Cl)(Cl)Cl", "1H", "no hydrogen"),
    ],
)
def test_predict_refuses_before_sending(smiles, nucleus, message):
    server = FakeServer()
    with pytest.raises(cc.CascadeError, match=message):
        cc.predict(Chem.MolFromSmiles(smiles), nucleus, opener=server)
    assert server.requests == []  # nothing left the machine


def test_predict_needs_a_molecule():
    with pytest.raises(cc.CascadeError, match="draw or load"):
        cc.predict(None, "1H", opener=FakeServer())


def test_unsanitizable_structure_is_reported():
    mol = Chem.RWMol(Chem.MolFromSmiles("C", sanitize=False))
    mol.GetAtomWithIdx(0).SetNumExplicitHs(5)
    with pytest.raises(cc.CascadeError, match="sanitized"):
        cc.prepare_molecule(mol)
