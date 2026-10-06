"""Dialog tests against real PyQt6, running offscreen.

The pattern to copy: build the dialog with fakes for the host callbacks, drive
the widgets, and monkeypatch the modal dialogs so nothing blocks the run.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_plugin.main_dialog import MyPluginDialog  # noqa: E402


class FakeMol:
    """Minimal stand-in for an RDKit molecule."""

    def __init__(self, count):
        self._count = count

    def GetNumAtoms(self):
        return self._count


@pytest.fixture
def dialog(qapp):
    dlg = MyPluginDialog(
        persistent_settings={"greeting": "Hi", "repeat": 2},
        get_molecule=lambda: FakeMol(3),
    )
    yield dlg
    dlg.deleteLater()


def test_settings_reach_the_widgets(dialog):
    assert dialog.greeting_edit.text() == "Hi"
    assert dialog.repeat_spin.value() == 2


def test_preview_follows_the_widgets(dialog):
    assert dialog.preview.toPlainText() == "Hi\nHi\n"
    dialog.repeat_spin.setValue(3)
    assert dialog.preview.toPlainText().count("Hi") == 3


def test_settings_roundtrip(dialog):
    dialog.apply_settings({"greeting": "Bonjour", "repeat": 5})
    assert dialog.read_settings() == {"greeting": "Bonjour", "repeat": 5}


def test_persistent_settings_are_updated(dialog):
    dialog.greeting_edit.setText("Hello")
    assert dialog.persistent_settings["greeting"] == "Hello"


def test_project_is_marked_modified(qapp):
    seen = []
    dlg = MyPluginDialog(persistent_settings={}, mark_modified=lambda: seen.append(1))
    dlg.repeat_spin.setValue(3)
    assert seen
    dlg.deleteLater()


def test_molecule_summary(dialog):
    assert "3 atoms" in dialog.summary_label.text()


def test_missing_molecule_is_reported(qapp):
    dlg = MyPluginDialog(persistent_settings={}, get_molecule=lambda: None)
    assert "No molecule" in dlg.summary_label.text()
    dlg.deleteLater()


def test_save_writes_the_file(dialog, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    target = tmp_path / "out.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dialog.save_text()
    text = target.read_text(encoding="utf-8")
    assert text == "Hi\nHi\n"
    assert "\r" not in text  # newline="\n" keeps it identical on every platform


def test_save_is_cancellable(dialog, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    dialog.save_text()
    assert not list(tmp_path.iterdir())


def test_save_reports_an_error(dialog, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    seen = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "no" / "out.txt"), "")
    )
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: seen.append(a))
    dialog.save_text()
    assert seen
