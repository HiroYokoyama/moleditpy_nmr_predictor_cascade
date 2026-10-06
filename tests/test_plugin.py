"""Metadata, registration and settings.json tests (no Qt needed)."""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nmr_predictor_cascade as plugin  # noqa: E402


class FakeContext:
    def __init__(self):
        self.menu_actions = []
        self.export_actions = []
        self.analysis_tools = []

    def add_menu_action(self, path, callback):
        self.menu_actions.append((path, callback))

    def add_export_action(self, label, callback):
        self.export_actions.append((label, callback))

    def add_analysis_tool(self, label, callback):
        self.analysis_tools.append((label, callback))


# -- metadata the registry reads ---------------------------------------------


def test_required_metadata_is_present():
    for name in (
        "PLUGIN_NAME",
        "PLUGIN_VERSION",
        "PLUGIN_AUTHOR",
        "PLUGIN_DESCRIPTION",
        "PLUGIN_SUPPORTED_MOLEDITPY_VERSION",
    ):
        assert getattr(plugin, name, "").strip(), f"{name} must be set"


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)


def test_description_discloses_the_external_server():
    """Users must learn from the Plugin Manager that the structure leaves the machine."""
    text = plugin.PLUGIN_DESCRIPTION.lower()
    assert "external" in text
    assert "nova.chem.colostate.edu" in text
    assert "internet" in text


def test_dependencies_are_lists_without_host_packages():
    assert isinstance(plugin.PLUGIN_DEPENDENCIES, list)
    assert isinstance(plugin.PLUGIN_OPTIONAL_DEPENDENCIES, list)
    for name in ("PyQt6", "rdkit", "numpy"):
        assert name not in plugin.PLUGIN_DEPENDENCIES
    assert "matplotlib" in plugin.PLUGIN_DEPENDENCIES


def test_tags_are_a_short_list():
    assert isinstance(plugin.PLUGIN_TAGS, list)
    assert 1 <= len(plugin.PLUGIN_TAGS) <= 3


def test_module_has_no_run_or_autorun():
    """A module-level run() would make the host add a second menu entry."""
    assert not hasattr(plugin, "run")
    assert not hasattr(plugin, "autorun")


def test_importing_the_package_needs_no_qt_or_rdkit():
    """The host imports __init__ at startup; Qt/RDKit work happens later."""
    import subprocess

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('PyQt6', 'rdkit', 'matplotlib', 'pyvista'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import nmr_predictor_cascade as p\n"
        "p.initialize(type('C', (), {'add_menu_action': lambda *a: None})())\n"
        "print(p.load_settings('does-not-exist.json')['nucleus'])\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1H"


# -- registration --------------------------------------------------------------


def test_initialize_registers_exactly_one_entry():
    ctx = FakeContext()
    plugin.initialize(ctx)
    entries = ctx.menu_actions + ctx.export_actions + ctx.analysis_tools
    assert len(entries) == 1
    assert ctx.menu_actions[0][0] == "Analysis/NMR Prediction (CASCADE)..."


# -- settings.json -------------------------------------------------------------


def test_load_settings_defaults_when_missing(tmp_path):
    settings = plugin.load_settings(str(tmp_path / "nope.json"))
    assert settings == plugin.get_default_settings()


def test_settings_round_trip(tmp_path):
    path = str(tmp_path / "settings.json")
    settings = plugin.get_default_settings()
    settings.update({"nucleus": "13C", "spectrometer_mhz": 600.0, "show_multiplets": False})
    plugin.save_settings(settings, path)
    assert plugin.load_settings(path) == settings
    with open(path, "rb") as handle:
        assert b"\r\n" not in handle.read()


def test_load_settings_ignores_junk(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert plugin.load_settings(str(path)) == plugin.get_default_settings()
    path.write_text(json.dumps({"nucleus": 5, "symmetrize": "yes", "extra": 1}), encoding="utf-8")
    assert plugin.load_settings(str(path)) == plugin.get_default_settings()


def test_load_settings_accepts_integer_frequency(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"spectrometer_mhz": 500}), encoding="utf-8")
    assert plugin.load_settings(str(path))["spectrometer_mhz"] == 500.0


def test_save_settings_survives_unwritable_path(tmp_path):
    plugin.save_settings({"a": 1}, str(tmp_path / "missing_dir" / "settings.json"))


@pytest.mark.parametrize("name", ["settings.json"])
def test_settings_file_lives_beside_the_package(name):
    assert os.path.dirname(plugin.SETTINGS_FILE) == os.path.dirname(os.path.abspath(plugin.__file__))
    assert os.path.basename(plugin.SETTINGS_FILE) == name
