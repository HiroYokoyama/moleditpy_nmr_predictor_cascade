"""Render the real Qt dialogs with fixed example shifts, without network access."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmr_predictor_cascade import coupling, get_default_settings  # noqa: E402
from nmr_predictor_cascade.result_dialog import ResultDialog  # noqa: E402
from nmr_predictor_cascade.ui import PredictDialog  # noqa: E402


def render():
    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Arial", 9))
    app.setStyle("Fusion")
    examples = [
        ("1H", "C/C=C/C(=O)OC", "main.png", None),
        ("13C", "Cc1ccccc1", "13c.png", [21.43, 137.82, 128.90, 128.33, 125.61, 128.33, 128.90]),
    ]
    for nucleus, smiles, filename, shifts in examples:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=203)
        data = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != ("H" if nucleus == "1H" else "C"):
                continue
            parent = atom.GetNeighbors()[0] if nucleus == "1H" else atom
            ppm = ({0: 2.06, 1: 7.19, 2: 5.88, 6: 3.55}[parent.GetIdx()]
                   if nucleus == "1H" else shifts[atom.GetIdx()])
            data.append(dict(idx=atom.GetIdx(), parent_idx=parent.GetIdx(),
                             atom=atom.GetSymbol(), ppm=ppm, confidence="High"))
        coupling.annotate_predictions(mol, data, nucleus)
        window = SimpleNamespace(current_mol=mol, plotter=MagicMock())
        context = SimpleNamespace(get_main_window=lambda: window)
        settings = {**get_default_settings(), "show_multiplets": nucleus == "1H"}
        dialog = ResultDialog(None, dict(nucleus=nucleus, data=data, mol_with_h=mol), context, settings)
        dialog.auto_scale_chk.setChecked(True)
        dialog.show()
        app.processEvents()
        dialog.canvas.draw()
        app.processEvents()
        dialog.grab().save(str(ROOT / "img" / filename))
        if nucleus == "1H":
            dialog.auto_scale_chk.setChecked(False)
            dialog.min_ppm_spin.setValue(5.6)
            dialog.max_ppm_spin.setValue(7.5)
            app.processEvents()
            dialog.canvas.draw()
            app.processEvents()
            dialog.canvas.grab().save(str(ROOT / "img" / "multiplets.png"))
            confirm = PredictDialog(None, settings, smiles)
            confirm.show()
            app.processEvents()
            confirm.grab().save(str(ROOT / "img" / "confirm.png"))
            confirm.close()
        dialog.close()
        app.processEvents()


if __name__ == "__main__":
    render()
