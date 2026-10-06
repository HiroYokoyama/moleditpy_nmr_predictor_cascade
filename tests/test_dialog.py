"""Dialog tests against real PyQt6 + matplotlib, offscreen.

Skipped when PyQt6, matplotlib or RDKit are missing. pyvista is replaced by a
stub when it is not installed — only Sphere() is used, and the plotter is a
MagicMock anyway.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("matplotlib")
pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

if "pyvista" not in sys.modules:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        sys.modules["pyvista"] = types.SimpleNamespace(Sphere=lambda **kw: kw)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmr_predictor_cascade import coupling  # noqa: E402
from nmr_predictor_cascade import result_dialog as rd  # noqa: E402
from nmr_predictor_cascade.ui import PredictDialog  # noqa: E402


def _ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


def _result(nucleus):
    mol = _ethanol()
    if nucleus == "1H":
        data = [
            {"idx": i, "parent_idx": 0, "atom": "H", "ppm": 1.29, "confidence": "High"}
            for i in (3, 4, 5)
        ] + [
            {"idx": i, "parent_idx": 1, "atom": "H", "ppm": 3.76, "confidence": "High"}
            for i in (6, 7)
        ]
    else:
        data = [
            {"idx": 0, "parent_idx": 0, "atom": "C", "ppm": 18.22, "confidence": "Moderate"},
            {"idx": 1, "parent_idx": 1, "atom": "C", "ppm": 58.14, "confidence": "High"},
        ]
    coupling.annotate_predictions(mol, data, nucleus)
    return {"nucleus": nucleus, "data": data, "mol_with_h": mol, "n_conformers": 2}, mol


class FakeContext:
    def __init__(self, mol):
        self.mw = MagicMock()
        self.mw.current_mol = mol
        self.mw.edit_3d_manager.selected_atoms_3d = set()

    def get_main_window(self):
        return self.mw


@pytest.fixture
def make_dialog(qapp):
    made = []

    def factory(nucleus, settings=None):
        result, mol = _result(nucleus)
        ctx = FakeContext(mol)
        dlg = rd.ResultDialog(None, result, ctx, settings or {"spectrometer_mhz": 400.0})
        made.append(dlg)
        return dlg, ctx

    yield factory
    for dlg in made:
        dlg.sel_timer.stop()
        dlg.deleteLater()


# -- pure helpers --------------------------------------------------------------


def test_columns_and_rows():
    item = {"idx": 1, "atom": "C", "ppm": 58.144, "mult": "t", "j_text": "141", "confidence": "High", "dept": "CH2"}
    assert rd.row_values(item, "13C") == ["1", "CH2", "58.14", "t", "141", "High"]
    assert rd.columns_for("1H")[3:5] == ["Mult.", "J (Hz)"]
    assert "1J(CH) (Hz)" in rd.columns_for("13C")


def test_observe_frequency():
    assert rd.observe_mhz(400.0, "1H") == 400.0
    assert rd.observe_mhz(400.0, "13C") == pytest.approx(100.58, abs=0.01)


def test_nearest_peak():
    assert rd.nearest_peak([1.0, 3.0], 2.9, 0.2) == 1
    assert rd.nearest_peak([1.0, 3.0], 2.0, 0.2) is None
    assert rd.nearest_peak([], 2.0, 0.2) is None


# -- the result window -----------------------------------------------------------


def test_1h_table_shows_multiplets(make_dialog):
    dlg, _ = make_dialog("1H")
    assert dlg.table.rowCount() == 5
    assert dlg.table.columnCount() == 6
    rows = {dlg.table.item(r, 0).text(): (dlg.table.item(r, 3).text(), dlg.table.item(r, 4).text()) for r in range(5)}
    assert rows["3"] == ("t", "7.0")
    assert rows["6"] == ("q", "7.0")


def test_13c_table_shows_carbon_types(make_dialog):
    dlg, _ = make_dialog("13C")
    assert [dlg.table.item(r, 1).text() for r in range(2)] == ["CH3", "CH2"]
    assert [dlg.table.item(r, 4).text() for r in range(2)] == ["125", "141"]


def test_multiplet_toggle_changes_the_sticks(make_dialog):
    dlg, _ = make_dialog("1H", {"show_multiplets": True, "spectrometer_mhz": 400.0})
    split = dlg.sticks()
    assert len(split) == 3 + 4  # triplet + quartet
    dlg.multiplet_chk.setChecked(False)
    assert dlg.sticks() == [(1.29, 3.0), (3.76, 2.0)]
    assert dlg.settings["show_multiplets"] is False


def test_spectrometer_frequency_narrows_the_multiplet(make_dialog):
    dlg, _ = make_dialog("1H")
    width = lambda sticks: max(p for p, _ in sticks if p < 2) - min(p for p, _ in sticks if p < 2)
    at_400 = width(dlg.sticks())
    dlg.mhz_spin.setValue(800.0)
    assert width(dlg.sticks()) == pytest.approx(at_400 / 2)


def test_broadening_is_on_by_default(make_dialog):
    dlg, _ = make_dialog("1H")
    assert dlg.broadening_chk.isChecked()
    assert dlg.linewidth_spin.value() == 1.0
    lines = dlg.figure.axes[0].get_lines()
    assert any(len(line.get_xdata()) > 100 for line in lines)  # a curve, not sticks
    dlg13, _ = make_dialog("13C")
    assert dlg13.linewidth_spin.value() == 2.0


def test_broadened_curve_follows_the_axis_range(make_dialog):
    dlg, _ = make_dialog("1H")
    x, y = dlg.curve()
    assert min(x) == pytest.approx(-1.0) and max(x) == pytest.approx(12.0)
    assert max(y) == pytest.approx(1.5, abs=0.01)  # centre line of the CH3 triplet: 3 x 0.5


def test_broadening_can_be_switched_off(make_dialog):
    dlg, _ = make_dialog("1H")
    dlg.broadening_chk.setChecked(False)
    assert dlg.settings["broadening"] is False
    assert not dlg.linewidth_spin.isEnabled()
    ax = dlg.figure.axes[0]
    assert not any(len(line.get_xdata()) > 100 for line in ax.get_lines())
    assert ax.collections  # the vlines


def test_broadening_off_from_settings(make_dialog):
    dlg, _ = make_dialog("1H", {"broadening": False, "spectrometer_mhz": 400.0})
    assert not dlg.broadening_chk.isChecked()


def test_about_shows_the_shared_module_version(make_dialog, monkeypatch):
    shown = {}

    def fake_exec(box):
        shown["text"] = box.text()

    monkeypatch.setattr(rd.QMessageBox, "exec", fake_exec)
    dlg, _ = make_dialog("1H")
    dlg.show_about()
    assert f"Shared coupling module: {coupling.COUPLING_MODULE_VERSION}" in shown["text"]


def test_highlight_adds_and_clears_actors(make_dialog):
    dlg, ctx = make_dialog("1H")
    dlg.highlight_atom(0, persistent=True)
    plotter = ctx.mw.plotter
    assert plotter.add_mesh.call_count == 3  # the three equivalent methyl protons
    for call in plotter.add_mesh.call_args_list:
        assert call.kwargs["reset_camera"] is False
    assert "3 atoms" in dlg.status_label.text()
    dlg.clear_selection()
    assert plotter.remove_actor.call_count == 6  # three spheres + three labels
    assert dlg._persistent_ppm is None


def test_highlight_falls_back_to_the_heavy_atom(make_dialog):
    """A host molecule with implicit H has no atom 6; highlight carbon 1 instead."""
    dlg, ctx = make_dialog("1H")
    heavy = Chem.RemoveHs(ctx.mw.current_mol)
    ctx.mw.current_mol = heavy
    dlg.highlight_atom(3, persistent=True)
    centers = [c.kwargs["name"] for c in ctx.mw.plotter.add_mesh.call_args_list]
    assert centers == ["cascade_nmr_highlight_1"]


def test_sync_from_3d_selects_the_row(make_dialog):
    dlg, ctx = make_dialog("13C")
    ctx.mw.edit_3d_manager.selected_atoms_3d = {1}
    dlg._sync_from_3d()
    assert dlg._persistent_ppm == 58.14


def test_csv_export(tmp_path):
    result, _ = _result("13C")
    path = tmp_path / "out.csv"
    rd.write_csv(str(path), result["data"], "13C")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Atom ID,Type,Shift (ppm)")
    assert lines[1] == "0,CH3,18.22,q,125,Moderate"


# -- the confirmation dialog -------------------------------------------------------


def test_predict_dialog_shows_what_is_sent(qapp):
    dlg = PredictDialog(None, {"nucleus": "13C", "spectrometer_mhz": 500.0, "symmetrize": False,
                               "server": "https://nova.chem.colostate.edu"}, "CCO")
    assert dlg.smiles_edit.text() == "CCO"
    assert dlg.smiles_edit.isReadOnly()
    assert dlg.nucleus_combo.currentText() == "13C"
    assert dlg.send_button.text() == "Send && Predict"
    dlg.nucleus_combo.setCurrentText("1H")
    dlg.mhz_spin.setValue(600.0)
    chosen = dlg.chosen_settings()
    assert chosen["nucleus"] == "1H" and chosen["spectrometer_mhz"] == 600.0
    assert chosen["symmetrize"] is False
    dlg.deleteLater()
