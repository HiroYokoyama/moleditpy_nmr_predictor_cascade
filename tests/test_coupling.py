"""Tests for the shared coupling module.

SHARED TEST FILE — identical in moleditpy_nmr_predictor_cascade and
moleditpy_nmr_predicator_nmrshiftdb2; only the import below differs.

Needs RDKit (skipped without it). Molecules are embedded with a fixed seed
so the Karplus terms are reproducible.
"""

import importlib.util
import os

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

# Load coupling.py by path: importing the package would pull in Qt.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = next(
    d
    for d in os.listdir(_ROOT)
    if os.path.isfile(os.path.join(_ROOT, d, "coupling.py"))
)
_spec = importlib.util.spec_from_file_location(
    "shared_coupling", os.path.join(_ROOT, _PKG, "coupling.py")
)
coupling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coupling)


def _mol(smiles, embed=True):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if embed:
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _by_carbon(mol):
    """{carbon index: (mult, j_text)} for the protons on each carbon."""
    info = coupling.hh_multiplets(mol, coupling.predict_hh_couplings(mol))
    out = {}
    for h_idx, value in info.items():
        parent = mol.GetAtomWithIdx(h_idx).GetNeighbors()[0].GetIdx()
        out.setdefault(parent, set()).add((value["mult"], value["j_text"]))
    return out


# -- small pure functions ----------------------------------------------------


def test_karplus_shape():
    assert coupling.karplus(180) == pytest.approx(10.26, abs=0.01)
    assert coupling.karplus(90) == pytest.approx(1.40, abs=0.01)
    assert coupling.karplus(0) == pytest.approx(8.06, abs=0.01)


@pytest.mark.parametrize(
    "pattern, name",
    [
        ([], "s"),
        ([(7.0, 1)], "d"),
        ([(7.0, 2)], "t"),
        ([(7.0, 3)], "q"),
        ([(7.0, 6)], "sept"),
        ([(7.0, 7)], "m"),
        ([(10.0, 1), (2.0, 1)], "dd"),
        ([(17.0, 1), (7.0, 3)], "dq"),
        ([(7.0, 1), (5.0, 1), (3.0, 1)], "ddd"),
        ([(9.0, 1), (7.0, 1), (5.0, 1), (3.0, 1)], "m"),
        ([(7.0, 4), (2.0, 1)], "m"),
    ],
)
def test_multiplicity_name(pattern, name):
    assert coupling.multiplicity_name(pattern) == name


def test_merge_pattern_merges_close_couplings():
    assert coupling.merge_pattern([(7.1, 1), (6.9, 1)]) == [(7.0, 2)]
    assert coupling.merge_pattern([(2.0, 1), (10.0, 1)]) == [(10.0, 1), (2.0, 1)]


def test_multiplet_lines_quartet():
    lines = coupling.multiplet_lines(1.0, [(8.0, 3)], 400.0)
    assert [round(p, 4) for p, _ in lines] == [0.97, 0.99, 1.01, 1.03]
    assert [h for _, h in lines] == pytest.approx([0.125, 0.375, 0.375, 0.125])
    assert sum(h for _, h in lines) == pytest.approx(1.0)


def test_multiplet_lines_singlet():
    assert coupling.multiplet_lines(2.5, [], 400.0) == [(2.5, 1.0)]


# -- 1H couplings ------------------------------------------------------------


def test_ethanol_ethyl_group():
    by_c = _by_carbon(_mol("CCO"))
    assert by_c[0] == {("t", "7.0")}  # CH3
    assert by_c[1] == {("q", "7.0")}  # CH2 (OH exchanges, no coupling)


def test_hydroxyl_proton_is_not_coupled():
    mol = _mol("CCO")
    pairs = coupling.predict_hh_couplings(mol)
    oh = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetSymbol() == "O"]
    assert oh and not any(c["h1"] in oh or c["h2"] in oh for c in pairs)
    assert oh[0] not in coupling.hh_multiplets(mol, pairs)


def test_trans_alkene():
    by_c = _by_carbon(_mol("C/C=C/C(=O)OC"))
    assert by_c[1] == {("dq", "17.0, 7.0")}
    assert by_c[2] == {("d", "17.0")}
    assert by_c[6] == {("s", "")}  # OCH3


def test_cis_alkene():
    by_c = _by_carbon(_mol("C/C=C\\C(=O)OC"))
    assert by_c[2] == {("d", "10.0")}


def test_terminal_vinyl_protons_stay_distinct():
    """Cis and trans =CH2 protons must not be averaged into 13.5 Hz."""
    by_c = _by_carbon(_mol("C=CC"))
    assert by_c[0] == {("dd", "10.0, 2.0"), ("dd", "17.0, 2.0")}
    assert by_c[1] == {("ddq", "17.0, 10.0, 7.0")}


def test_aromatic_ortho_and_meta():
    by_c = _by_carbon(_mol("c1ccccc1C"))
    assert by_c[2] == {("tt", "7.5, 1.5")}  # para H: two ortho, two meta
    assert by_c[6] == {("s", "")}  # benzylic CH3 is not coupled to the ring


def test_benzene_is_a_singlet():
    by_c = _by_carbon(_mol("c1ccccc1"))
    assert all(v == {("s", "")} for v in by_c.values())


def test_aldehyde_small_coupling():
    by_c = _by_carbon(_mol("CC=O"))
    assert by_c[0] == {("d", "2.5")}
    assert by_c[1] == {("q", "2.5")}


def test_equivalent_ring_protons_share_a_multiplet():
    """Class averaging: every alpha proton of cyclohexanone gets one pattern."""
    by_c = _by_carbon(_mol("O=C1CCCCC1"))
    assert len(by_c[2] | by_c[6]) == 1


def test_works_without_3d():
    by_c = _by_carbon(_mol("CCO", embed=False))
    assert by_c[1] == {("q", "7.0")}


# -- 13C one-bond couplings --------------------------------------------------


@pytest.mark.parametrize(
    "smiles, idx, mult, j",
    [
        ("CC", 0, "q", 125.0),
        ("C", 0, "quint", 125.0),
        ("CO", 0, "q", 141.0),
        ("ClCCl", 1, "t", 175.0),
        ("c1ccccc1", 0, "d", 159.0),
        ("C=C", 0, "t", 157.0),
        ("C#C", 0, "d", 249.0),
        ("CC=O", 1, "d", 172.0),
        ("CC(C)(C)C", 1, "s", None),
    ],
)
def test_one_bond_ch(smiles, idx, mult, j):
    info = coupling.predict_ch_couplings(Chem.AddHs(Chem.MolFromSmiles(smiles)))[idx]
    assert info["mult"] == mult
    assert info["j_ch"] == j


def test_dept_labels():
    info = coupling.predict_ch_couplings(Chem.MolFromSmiles("CC(C)Cc1ccccc1"))
    assert info[0]["dept"] == "CH3"
    assert info[1]["dept"] == "CH"
    assert info[3]["dept"] == "CH2"
    assert info[4]["dept"] == "C"


# -- the one-call API ----------------------------------------------------------


def test_annotate_predictions_1h():
    mol = _mol("CCO")
    items = [{"idx": a.GetIdx(), "atom": "H", "ppm": 1.0} for a in mol.GetAtoms() if a.GetSymbol() == "H"]
    coupling.annotate_predictions(mol, items, "1H")
    mults = {i["mult"] for i in items}
    assert {"t", "q", ""} <= mults | {""}
    assert all("pattern" in i and "j_text" in i for i in items)


def test_annotate_predictions_13c():
    mol = _mol("CCO")
    items = [{"idx": 0, "atom": "C", "ppm": 18.0}, {"idx": 1, "atom": "C", "ppm": 58.0}]
    coupling.annotate_predictions(mol, items, "13C")
    assert [(i["mult"], i["dept"], i["j_ch"]) for i in items] == [
        ("q", "CH3", 125.0),
        ("t", "CH2", 141.0),
    ]


def test_annotate_predictions_never_raises():
    items = [{"idx": 0, "atom": "H", "ppm": 1.0}]
    coupling.annotate_predictions(None, items, "1H")
    assert items[0]["mult"] == "" and items[0]["pattern"] == []


def test_spectrum_sticks():
    items = [
        {"ppm": 1.2, "pattern": [(7.0, 2)]},
        {"ppm": 1.2, "pattern": [(7.0, 2)]},
        {"ppm": 3.5, "pattern": []},
    ]
    plain = coupling.spectrum_sticks(items, 400, show_multiplets=False)
    assert plain == [(1.2, 2.0), (3.5, 1.0)]
    split = coupling.spectrum_sticks(items, 400, show_multiplets=True)
    assert len(split) == 4  # triplet (3 lines) + singlet
    assert sum(h for _, h in split) == pytest.approx(3.0)
