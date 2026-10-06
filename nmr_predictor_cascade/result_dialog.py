"""Result window: stick spectrum, shift table and 3D highlighting.

Modelled on the nmrshiftdb2 predictor's result window, with the coupling
columns and the multiplet display added.
"""

import csv
import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import PLUGIN_AUTHOR, PLUGIN_VERSION
from . import coupling

#: 13C observe frequency relative to 1H (gyromagnetic ratio).
C13_TO_H1 = 0.25145

#: Room left beside the outermost peaks by "Auto Fit". 1 ppm is a lot of a
#: 1H axis but nothing on a 13C one, where it put the end peaks on the frame.
AUTO_FIT_MARGIN_PPM = {"1H": 1.0, "13C": 10.0}

#: Two peaks closer than this (ppm) are the same signal.
SAME_PEAK_PPM = 1e-4


def columns_for(nucleus):
    """Table headers; also the CSV header."""
    if nucleus == "1H":
        return ["Atom ID", "Type", "Shift (ppm)", "Mult.", "J (Hz)", "Confidence"]
    return ["Atom ID", "Type", "Shift (ppm)", "Mult. (1H-coupled)", "1J(CH) (Hz)", "Confidence"]


def row_values(item, nucleus):
    """The table/CSV cells of one prediction, as strings."""
    kind = item["atom"]
    if nucleus == "13C" and item.get("dept"):
        kind = item["dept"]  # C / CH / CH2 / CH3
    return [
        str(item["idx"]),
        kind,
        f"{item['ppm']:.2f}",
        item.get("mult", ""),
        item.get("j_text", ""),
        item.get("confidence", ""),
    ]


def observe_mhz(h1_mhz, nucleus):
    """Observe frequency of ``nucleus`` on a magnet with the given 1H frequency."""
    return h1_mhz if nucleus == "1H" else h1_mhz * C13_TO_H1


def nearest_peak(shifts, x, tolerance):
    """Index of the shift closest to ``x`` if within ``tolerance``, else None."""
    if not shifts or x is None:
        return None
    best = min(range(len(shifts)), key=lambda i: abs(shifts[i] - x))
    return best if abs(shifts[best] - x) < tolerance else None


class ResultDialog(QDialog):
    def __init__(self, parent, result_data, context, settings=None):
        super().__init__(parent)
        # matplotlib is imported here so that importing the package never needs it.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.context = context
        self.settings = dict(settings or {})
        self.data = result_data["data"]
        self.nucleus = result_data["nucleus"]
        self.result_info = result_data

        self.setWindowTitle(f"CASCADE NMR Prediction ({self.nucleus})")
        self.resize(680, 840)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self._highlight_names = []
        self._last_selected = set()
        self._hover_idx = -1
        self._persistent_ppm = None
        self._graph_line = None
        self._hover_line = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # -- spectrum ------------------------------------------------------
        self.figure = Figure(figsize=(5, 3.5))
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        title = QLabel("Graph Controls")
        title.setStyleSheet("font-weight: bold; color: #555;")

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range (ppm):"))
        self.min_ppm_spin = self._ppm_spin(-1.0 if self.nucleus == "1H" else -10.0)
        range_row.addWidget(self.min_ppm_spin)
        range_row.addWidget(QLabel("to"))
        self.max_ppm_spin = self._ppm_spin(12.0 if self.nucleus == "1H" else 220.0)
        range_row.addWidget(self.max_ppm_spin)
        self.auto_scale_chk = QCheckBox("Auto Fit")
        self.auto_scale_chk.toggled.connect(self.plot_spectrum)
        range_row.addWidget(self.auto_scale_chk)
        range_row.addStretch()

        mult_row = QHBoxLayout()
        self.multiplet_chk = QCheckBox(
            "Show multiplets" if self.nucleus == "1H" else "Show 1H-coupled multiplets"
        )
        self.multiplet_chk.setChecked(bool(self.settings.get("show_multiplets", True)))
        self.multiplet_chk.toggled.connect(self._on_multiplet_toggled)
        mult_row.addWidget(self.multiplet_chk)
        mult_row.addWidget(QLabel("Spectrometer:"))
        self.mhz_spin = QDoubleSpinBox()
        self.mhz_spin.setRange(40.0, 1200.0)
        self.mhz_spin.setDecimals(0)
        self.mhz_spin.setSingleStep(100.0)
        self.mhz_spin.setSuffix(" MHz (1H)")
        self.mhz_spin.setValue(float(self.settings.get("spectrometer_mhz", 400.0)))
        self.mhz_spin.valueChanged.connect(self.plot_spectrum)
        mult_row.addWidget(self.mhz_spin)
        mult_row.addStretch()

        broad_row = QHBoxLayout()
        self.broadening_chk = QCheckBox("Line broadening")
        self.broadening_chk.setChecked(bool(self.settings.get("broadening", True)))
        self.broadening_chk.setToolTip("Draw Lorentzian lines instead of sticks.")
        self.broadening_chk.toggled.connect(self._on_broadening_toggled)
        broad_row.addWidget(self.broadening_chk)
        broad_row.addWidget(QLabel("Line width:"))
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 50.0)
        self.linewidth_spin.setDecimals(1)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setSuffix(" Hz")
        self.linewidth_spin.setValue(coupling.DEFAULT_LINEWIDTH_HZ.get(self.nucleus, 1.0))
        self.linewidth_spin.valueChanged.connect(self.plot_spectrum)
        broad_row.addWidget(self.linewidth_spin)
        broad_row.addStretch()

        layout.addWidget(self.toolbar)
        layout.addWidget(title)
        layout.addLayout(range_row)
        layout.addLayout(mult_row)
        layout.addLayout(broad_row)
        layout.addWidget(self.canvas)

        # -- table ---------------------------------------------------------
        headers = columns_for(self.nucleus)
        self.table = QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(
            "QTableWidget::item:selected { background-color: pink; color: black; }"
        )
        self.table.setRowCount(len(self.data))
        for row, item in enumerate(self.data):
            for col, text in enumerate(row_values(item, self.nucleus)):
                cell = QTableWidgetItem(text)
                align = Qt.AlignmentFlag.AlignCenter
                if col == 2:
                    align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                cell.setTextAlignment(align)
                self.table.setItem(row, col, cell)
        layout.addWidget(self.table)

        note = QLabel(
            "Couplings are rule-based estimates (typical values and a Karplus curve), "
            "not part of the CASCADE prediction."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        self.status_label = QLabel("Hover or click peaks to see details.")
        self.status_label.setStyleSheet("color: #444; font-weight: bold;")
        layout.addWidget(self.status_label)

        # -- buttons -------------------------------------------------------
        buttons = QHBoxLayout()
        self.unselect_btn = QPushButton("Unselect All")
        self.unselect_btn.clicked.connect(self.clear_selection)
        buttons.addWidget(self.unselect_btn)
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_csv)
        buttons.addWidget(export_btn)
        buttons.addStretch()
        about_btn = QPushButton("About")
        about_btn.clicked.connect(self.show_about)
        buttons.addWidget(about_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        credit_row = QHBoxLayout()
        credit_row.addStretch()
        credit = QLabel("POWERED BY CASCADE (Paton group, Colorado State University)")
        credit.setStyleSheet("color: gray; font-size: 9px; font-style: italic;")
        credit_row.addWidget(credit)
        layout.addLayout(credit_row)

        self.plot_spectrum()

        self.table.cellClicked.connect(self.on_table_click)
        self.canvas.mpl_connect("button_press_event", self.on_graph_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # 3D selection -> table sync.
        self.sel_timer = QTimer(self)
        self.sel_timer.timeout.connect(self._sync_from_3d)
        self.sel_timer.start(300)

    def _ppm_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setRange(-50, 500)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.valueChanged.connect(self.plot_spectrum)
        return spin

    # -- spectrum --------------------------------------------------------

    def _on_multiplet_toggled(self, checked):
        self.settings["show_multiplets"] = bool(checked)
        self.plot_spectrum()

    def _on_broadening_toggled(self, checked):
        self.settings["broadening"] = bool(checked)
        self.linewidth_spin.setEnabled(bool(checked))
        self.plot_spectrum()

    def x_range(self):
        """(left, right) of the axis: high ppm on the left, as in NMR."""
        if self.auto_scale_chk.isChecked():
            shifts = [item["ppm"] for item in self.data]
            margin = AUTO_FIT_MARGIN_PPM.get(self.nucleus, 1.0)
            return max(shifts) + margin, min(shifts) - margin
        return self.max_ppm_spin.value(), self.min_ppm_spin.value()

    def curve(self):
        """Broadened spectrum ``(x, y)`` over the current axis range."""
        left, right = self.x_range()
        return coupling.lorentzian_curve(
            self.sticks(),
            self.linewidth_spin.value(),
            observe_mhz(self.mhz_spin.value(), self.nucleus),
            right,
            left,
        )

    def sticks(self):
        mhz = observe_mhz(self.mhz_spin.value(), self.nucleus)
        return coupling.spectrum_sticks(self.data, mhz, self.multiplet_chk.isChecked())

    def plot_spectrum(self, *_args):
        self.figure.clear()
        # figure.clear() destroyed the highlight artists.
        self._graph_line = None
        self._hover_line = None
        ax = self.figure.add_subplot(111)

        if not self.data:
            ax.text(0.5, 0.5, "No peaks predicted", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        x_values, y_values = [], []
        if self.broadening_chk.isChecked():
            x_values, y_values = self.curve()
        if x_values:
            ax.plot(x_values, y_values, color="b", linewidth=1.0)
            top = max(y_values)
        else:  # sticks, or no line inside the axis range
            sticks = self.sticks()
            heights = [h for _p, h in sticks]
            ax.vlines([p for p, _h in sticks], 0, heights, colors="b", linewidth=1.2)
            top = max(heights)
        ax.axhline(0, color="k", alpha=0.3, linewidth=1)
        ax.set_ylim(0, (top or 1.0) * 1.2)
        ax.set_xlim(*self.x_range())

        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"{self.nucleus} NMR Predicted Spectrum (CASCADE)")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        # Keep the selection marker across redraws.
        if self._persistent_ppm is not None:
            self._graph_line = ax.axvline(
                self._persistent_ppm, color="red", linestyle="-", alpha=0.8, linewidth=2
            )
        self.canvas.draw()

    def _tolerance(self, axes):
        xlim = axes.get_xlim()
        return abs(xlim[1] - xlim[0]) * 0.02

    def on_hover(self, event):
        if event.inaxes is None:
            if self._hover_idx != -1:
                self._hover_idx = -1
                self._update_graph_highlight(None, is_hover=True)
                self._restore_persistent_highlight()
            return

        shifts = [item["ppm"] for item in self.data]
        idx = nearest_peak(shifts, event.xdata, self._tolerance(event.inaxes))
        if idx is None:
            if self._hover_idx != -1:
                self._hover_idx = -1
                self._update_graph_highlight(None, is_hover=True)
                self._restore_persistent_highlight()
            return
        if idx == self._hover_idx:
            return
        self._hover_idx = idx
        target = shifts[idx]
        if self._persistent_ppm is not None and abs(target - self._persistent_ppm) < SAME_PEAK_PPM:
            self._update_graph_highlight(None, is_hover=True)  # already selected
            return
        self._update_graph_highlight(target, is_hover=True)
        self.highlight_atom(idx, persistent=False)
        self.status_label.setText(f"Peak: {self.describe(self.data[idx])}")
        self.status_label.setStyleSheet("color: #e67e22; font-weight: bold;")

    def describe(self, item):
        text = f"{item['atom']}{item['idx']} at {item['ppm']:.2f} ppm"
        if item.get("mult"):
            text += f", {item['mult']}"
            if item.get("j_text"):
                text += f" (J = {item['j_text']} Hz)"
        if item.get("confidence"):
            text += f", confidence {item['confidence']}"
        return text

    def _restore_persistent_highlight(self):
        if self._persistent_ppm is not None:
            for i, item in enumerate(self.data):
                if abs(item["ppm"] - self._persistent_ppm) < SAME_PEAK_PPM:
                    self.highlight_atom(i, persistent=True)
                    return
        self.clear_3d_visuals()
        self.status_label.setText("Hover over peaks to see in 3D.")
        self.status_label.setStyleSheet("color: #444; font-weight: bold;")

    def on_graph_click(self, event):
        if event.inaxes is None:
            return
        shifts = [item["ppm"] for item in self.data]
        idx = nearest_peak(shifts, event.xdata, self._tolerance(event.inaxes))
        if idx is None:
            self.clear_selection()
            return
        ppm = shifts[idx]
        self.table.clearSelection()
        self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        for row, item in enumerate(self.data):
            if abs(item["ppm"] - ppm) < SAME_PEAK_PPM:
                self.table.selectRow(row)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.highlight_atom(idx, persistent=True)

    def on_table_click(self, row, _col):
        self.highlight_atom(row)

    def _update_graph_highlight(self, ppm, is_hover=False):
        if not self.figure.axes:
            return
        ax = self.figure.axes[0]
        if self._hover_line is not None:
            try:
                self._hover_line.remove()
            except Exception:
                pass  # removed with the axes already
            self._hover_line = None
        if is_hover:
            if ppm is not None:
                self._hover_line = ax.axvline(ppm, color="orange", linestyle="--", alpha=0.5, linewidth=2)
        else:
            if self._graph_line is not None:
                try:
                    self._graph_line.remove()
                except Exception:
                    pass
                self._graph_line = None
            if ppm is not None:
                self._graph_line = ax.axvline(ppm, color="red", linestyle="-", alpha=0.8, linewidth=2)
        self.canvas.draw()

    # -- 3D highlighting -------------------------------------------------

    def highlight_atom(self, row_idx, persistent=True):
        """Sphere + label on every atom of the signal at this row's shift."""
        import pyvista as pv
        from rdkit import Chem

        target = self.data[row_idx]
        matching = [i for i in self.data if abs(i["ppm"] - target["ppm"]) < SAME_PEAK_PPM]
        try:
            mw = self.context.get_main_window()
            plotter = mw.plotter
            mol = mw.current_mol
            self.clear_3d_visuals()
            if mol is None or mol.GetNumConformers() == 0:
                self.status_label.setText("Convert the structure to 3D to see highlights.")
                return
            conf = mol.GetConformer()
            n_atoms = mol.GetNumAtoms()
            table = Chem.GetPeriodicTable()

            # A hydrogen index can sit past the end of a host molecule that
            # keeps its H implicit; fall back to the heavy atom it hangs off.
            # Equivalent protons of one CH2/CH3 then share a carbon: draw it once.
            targets = {}
            for item in matching:
                idx = item["idx"] if item["idx"] < n_atoms else item.get("parent_idx", item["idx"])
                if idx >= n_atoms:
                    self.status_label.setText(
                        "Structure changed since prediction - re-run the prediction."
                    )
                    return
                targets.setdefault(idx, item)
            targets = list(targets.items())

            color = "red" if persistent else "orange"
            for atom_idx, item in targets:
                pos = conf.GetAtomPosition(atom_idx)
                point = (pos.x, pos.y, pos.z)
                radius = table.GetRvdw(mol.GetAtomWithIdx(atom_idx).GetSymbol()) * 0.3 * 1.4
                sphere_name = f"cascade_nmr_highlight_{atom_idx}"
                label_name = f"cascade_nmr_label_{atom_idx}"
                plotter.add_mesh(
                    pv.Sphere(radius=radius, center=point),
                    color=color,
                    opacity=0.5 if persistent else 0.4,
                    name=sphere_name,
                    pickable=False,
                    reset_camera=False,  # a highlight must not undo the user's zoom
                )
                label = f"{item['atom']}{item['idx']}\n{item['ppm']:.2f}"
                if item.get("mult"):
                    label += f" {item['mult']}"
                plotter.add_point_labels(
                    [point],
                    [label],
                    font_size=12,
                    text_color="white" if persistent else "yellow",
                    point_size=0,
                    always_visible=True,
                    bold=True,
                    name=label_name,
                    reset_camera=False,
                )
                self._highlight_names += [sphere_name, label_name]

            if persistent:
                if len(matching) > 1:
                    self.status_label.setText(
                        f"Selected equivalent peak: {len(matching)} atoms, {self.describe(target)}"
                    )
                else:
                    self.status_label.setText(f"Selected {self.describe(target)}")
                self.status_label.setStyleSheet("color: #444; font-weight: bold;")
                self._persistent_ppm = target["ppm"]
                self._update_graph_highlight(target["ppm"])
            plotter.render()
        except Exception as exc:
            logging.warning("CASCADE NMR: highlight failed: %s", exc)

    def clear_3d_visuals(self):
        try:
            plotter = self.context.get_main_window().plotter
            for name in self._highlight_names:
                plotter.remove_actor(name)
            self._highlight_names = []
            plotter.render()
        except Exception as exc:
            logging.warning("CASCADE NMR: could not clear highlights: %s", exc)

    def clear_selection(self):
        self.table.clearSelection()
        self._persistent_ppm = None
        self._update_graph_highlight(None)
        self.clear_3d_visuals()
        self.status_label.setText("Selection cleared.")

    def _sync_from_3d(self):
        """Select the table row of an atom picked in the 3D view."""
        mw = self.context.get_main_window()
        manager = getattr(mw, "edit_3d_manager", None)
        selected = getattr(manager, "selected_atoms_3d", None)
        if selected is None:
            return
        current = set(selected)
        if current == self._last_selected:
            return
        self._last_selected = current
        if not current:
            self.clear_3d_visuals()
            self.table.clearSelection()
            return
        for row, item in enumerate(self.data):
            if item["idx"] in current:
                self.table.selectRow(row)
                self.highlight_atom(row)
                return

    # -- output ------------------------------------------------------------

    def export_csv(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", f"cascade_{self.nucleus}.csv", "CSV Files (*.csv)"
        )
        if not filename:
            return
        try:
            write_csv(filename, self.data, self.nucleus)
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to save file:\n{exc}")
            return
        QMessageBox.information(self, "Success", f"Exported successfully to:\n{filename}")

    def show_about(self):
        n_conf = self.result_info.get("n_conformers") or "?"
        text = f"""
        <h3>NMR Predictor (CASCADE)</h3>
        <p>Chemical shifts are predicted by <b>CASCADE</b>, a graph neural network on
        an MMFF conformer ensemble, run on the web server of the Paton group,
        Colorado State University. This prediction used {n_conf} conformer(s).<br>
        <a href="https://nova.chem.colostate.edu/v2/cascade/">https://nova.chem.colostate.edu/v2/cascade/</a></p>
        <p>Please cite: Guan, Y.; Sowndarya, S. V. S.; Gallegos, L. C.; St. John, P. C.;
        Paton, R. S. <i>Chem. Sci.</i> <b>2021</b>, 12, 12012-12026.</p>
        <p><b>Couplings</b> (multiplicity, J) are rule-based estimates made by this
        plugin from typical values and a Karplus curve; they are not part of CASCADE.</p>
        <p>Author: {PLUGIN_AUTHOR}<br>Version: {PLUGIN_VERSION}<br>
        Shared coupling module: {coupling.COUPLING_MODULE_VERSION}</p>
        """
        box = QMessageBox(self)
        box.setWindowTitle("About NMR Predictor (CASCADE)")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(text)
        box.exec()

    def closeEvent(self, event):
        self.clear_3d_visuals()
        self.sel_timer.stop()
        super().closeEvent(event)


def write_csv(path, data, nucleus):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns_for(nucleus))
        for item in data:
            writer.writerow(row_values(item, nucleus))
