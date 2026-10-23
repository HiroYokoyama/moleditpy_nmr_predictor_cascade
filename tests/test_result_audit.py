"""Malformed result, archive and closed-dialog regressions without networking."""

import io
import tarfile

import pytest
from rdkit import Chem

from nmr_predictor_cascade import cascade_client as cc
from nmr_predictor_cascade import coupling


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_nonfinite_shifts_are_skipped(value):
    rows = cc.parse_weighted_shifts(f"atom_index,Shift\n1,{value}\n2,3.5\n0,4\n")
    assert rows == [(1, 3.5, "")]


def _archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def _client_with(raw):
    return cc.CascadeClient(opener=lambda *args: io.BytesIO(raw))


def test_response_is_bounded(monkeypatch):
    monkeypatch.setattr(cc, "MAX_RESPONSE_BYTES", 16)
    with pytest.raises(cc.CascadeError, match="response exceeds"):
        _client_with(b"x" * 17).download("task")


def test_compressed_archive_is_bounded_after_decompression(monkeypatch):
    monkeypatch.setattr(cc, "MAX_ARCHIVE_BYTES", 1024)
    with pytest.raises(cc.CascadeError, match="unpacked size"):
        _client_with(_archive([("big.sdf", b"x" * 2048)])).download("task")


def test_duplicate_archive_basenames_are_rejected():
    raw = _archive(
        [("first/weighted_shift.csv", b"one"), ("second/weighted_shift.csv", b"two")]
    )
    with pytest.raises(cc.CascadeError, match="duplicate"):
        _client_with(raw).download("task")


def test_oversized_pax_metadata_is_bounded_before_tar_parsing(monkeypatch):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("weighted_shift.csv")
        member.pax_headers = {"comment": "x" * 40000}
        archive.addfile(member, io.BytesIO())
    monkeypatch.setattr(cc, "MAX_ARCHIVE_BYTES", 20000)
    with pytest.raises(cc.CascadeError, match="unpacked size"):
        _client_with(output.getvalue()).download("task")


def test_excessive_archive_members_are_rejected(monkeypatch):
    monkeypatch.setattr(cc, "MAX_ARCHIVE_MEMBERS", 2)
    with pytest.raises(cc.CascadeError, match="too many"):
        _client_with(_archive([(str(i), b"x") for i in range(3)])).download("task")


def test_missing_conformer_ids_do_not_count_as_a_conformer():
    assert cc.parse_conformer_count("atom_index,Shift\n1,2\n") == 0


def test_deuterium_is_not_counted_as_a_coupling_proton():
    mol = Chem.AddHs(Chem.MolFromSmiles("[2H]C([2H])([2H])C"))
    carbon = 1
    assert coupling.predict_ch_couplings(mol)[carbon]["n_h"] == 0
    assert all(
        not coupling._is_coupling_proton(a)
        for a in mol.GetAtoms()
        if a.GetIsotope() == 2
    )
