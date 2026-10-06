"""Menu flow: confirm what is sent, run the job off the UI thread, show the result."""

import logging

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
)

from . import PLUGIN_NAME, RESULT_WINDOW_ID, load_settings, save_settings
from . import cascade_client

# The running worker. Kept here, not on the main window, so the thread object
# outlives the menu callback that started it.
_worker = None


class PredictDialog(QDialog):
    """Nucleus choice plus an explicit statement of what leaves the computer."""

    def __init__(self, parent, settings, smiles):
        super().__init__(parent)
        self.setWindowTitle("NMR Prediction (CASCADE)")
        self.settings = dict(settings)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.nucleus_combo = QComboBox()
        self.nucleus_combo.addItems(["1H", "13C"])
        self.nucleus_combo.setCurrentText(self.settings.get("nucleus", "1H"))
        form.addRow("Nucleus:", self.nucleus_combo)

        self.mhz_spin = QDoubleSpinBox()
        self.mhz_spin.setRange(40.0, 1200.0)
        self.mhz_spin.setDecimals(0)
        self.mhz_spin.setSingleStep(100.0)
        self.mhz_spin.setSuffix(" MHz (1H)")
        self.mhz_spin.setValue(float(self.settings.get("spectrometer_mhz", 400.0)))
        self.mhz_spin.setToolTip("Spectrometer frequency used to draw the multiplets.")
        form.addRow("Spectrometer:", self.mhz_spin)

        self.symmetrize_check = QCheckBox("Average symmetry-equivalent atoms")
        self.symmetrize_check.setChecked(bool(self.settings.get("symmetrize", True)))
        form.addRow("", self.symmetrize_check)
        layout.addLayout(form)

        server = self.settings.get("server") or cascade_client.DEFAULT_SERVER
        notice = QLabel(
            "<b>This structure will be sent to an external server.</b><br>"
            f"Server: {server}<br>"
            "CASCADE is run by the Paton group, Colorado State University. "
            "Do not send structures you need to keep confidential."
        )
        notice.setWordWrap(True)
        notice.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(notice)

        layout.addWidget(QLabel("SMILES that will be sent:"))
        self.smiles_edit = QLineEdit(smiles)
        self.smiles_edit.setReadOnly(True)
        layout.addWidget(self.smiles_edit)

        buttons = QDialogButtonBox()
        self.send_button = buttons.addButton("Send && Predict", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def chosen_settings(self):
        self.settings.update(
            {
                "nucleus": self.nucleus_combo.currentText(),
                "spectrometer_mhz": float(self.mhz_spin.value()),
                "symmetrize": self.symmetrize_check.isChecked(),
            }
        )
        return self.settings


class PredictWorker(QThread):
    """Runs ``cascade_client.predict`` — network and RDKit work — off the UI thread."""

    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(self, mol, settings):
        super().__init__()
        self.mol = mol
        self.settings = dict(settings)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled

    def run(self):
        try:
            result = cascade_client.predict(
                self.mol,
                self.settings.get("nucleus", "1H"),
                server=self.settings.get("server"),
                is_cancelled=self.is_cancelled,
                progress=self.progress_signal.emit,
                symmetrize_shifts=self.settings.get("symmetrize", True),
            )
        except cascade_client.Cancelled:
            return
        except cascade_client.CascadeError as exc:
            self.error_signal.emit(str(exc))
            return
        except Exception as exc:  # anything unexpected still reaches the user
            logging.exception("%s: prediction failed", PLUGIN_NAME)
            self.error_signal.emit(f"Unexpected error: {exc}")
            return
        if not self._cancelled:
            self.finished_signal.emit(result)


def start_prediction(context):
    """Menu entry: confirm, predict in the background, open the result."""
    global _worker
    from rdkit import Chem

    mw = context.get_main_window()
    mol = context.current_molecule
    if mol is None or mol.GetNumAtoms() == 0:
        QMessageBox.warning(mw, PLUGIN_NAME, "Please draw or load a molecule first.")
        return
    if _worker is not None and _worker.isRunning():
        QMessageBox.information(mw, PLUGIN_NAME, "A CASCADE prediction is already running.")
        return

    try:
        smiles = cascade_client.to_smiles(cascade_client.prepare_molecule(mol))
    except cascade_client.CascadeError as exc:
        QMessageBox.warning(mw, PLUGIN_NAME, str(exc))
        return

    dialog = PredictDialog(mw, load_settings(), smiles)
    if not dialog.exec():
        return
    settings = dialog.chosen_settings()
    save_settings(settings)

    progress = QProgressDialog("Contacting the CASCADE server...", "Cancel", 0, 0, mw)
    progress.setWindowTitle(PLUGIN_NAME)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()

    worker = PredictWorker(Chem.Mol(mol), settings)
    _worker = worker

    def release():
        global _worker
        if _worker is worker:
            _worker = None
        worker.deleteLater()

    def on_progress(text):
        if not progress.wasCanceled():
            progress.setLabelText(text)

    def on_success(result):
        progress.cancel()
        show_result(context, result, settings)

    def on_error(message):
        cancelled = progress.wasCanceled()
        progress.cancel()
        if not cancelled:
            QMessageBox.critical(mw, "CASCADE Prediction Error", message)

    progress.canceled.connect(worker.cancel)
    worker.progress_signal.connect(on_progress)
    worker.finished_signal.connect(on_success)
    worker.error_signal.connect(on_error)
    worker.finished.connect(release)
    worker.start()


def show_result(context, result, settings):
    """Replace any open result window with a new one."""
    from .result_dialog import ResultDialog

    previous = context.get_window(RESULT_WINDOW_ID)
    if previous is not None:
        try:
            previous.close()
        except RuntimeError:
            pass  # already deleted on the C++ side
    dialog = ResultDialog(context.get_main_window(), result, context, settings)
    context.register_window(RESULT_WINDOW_ID, dialog)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
