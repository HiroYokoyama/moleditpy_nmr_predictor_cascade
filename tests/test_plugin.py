"""Metadata and registration tests — the checks every plugin should keep."""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import my_plugin as plugin  # noqa: E402


class FakeContext:
    """Stands in for the host's PluginContext."""

    def __init__(self, main_window=None):
        self.main_window = main_window
        self.menu_actions = []
        self.export_actions = []
        self.analysis_tools = []
        self.save_handlers = []
        self.load_handlers = []
        self.reset_handlers = []
        self.windows = {}
        self.current_molecule = None
        self.modified = 0

    def add_menu_action(self, path, callback):
        self.menu_actions.append((path, callback))

    def add_export_action(self, label, callback):
        self.export_actions.append((label, callback))

    def add_analysis_tool(self, label, callback):
        self.analysis_tools.append((label, callback))

    def register_save_handler(self, callback):
        self.save_handlers.append(callback)

    def register_load_handler(self, callback):
        self.load_handlers.append(callback)

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def register_window(self, window_id, window):
        self.windows[window_id] = window

    def get_window(self, window_id):
        return self.windows.get(window_id)

    def get_main_window(self):
        return self.main_window

    def mark_project_modified(self):
        self.modified += 1


@pytest.fixture
def context():
    original = dict(plugin.current_settings)
    ctx = FakeContext()
    plugin.initialize(ctx)
    yield ctx
    plugin._context = None
    plugin._dialog_opened = False
    plugin.current_settings.clear()
    plugin.current_settings.update(original)


# -- metadata the registry reads -------------------------------------------


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
    """The release tag is checked against this, so it must be vMAJOR.MINOR.PATCH."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)


def test_dependencies_are_a_list():
    assert isinstance(plugin.PLUGIN_DEPENDENCIES, list)
    assert isinstance(plugin.PLUGIN_OPTIONAL_DEPENDENCIES, list)
    for name in ("PyQt6", "rdkit", "numpy"):
        assert name not in plugin.PLUGIN_DEPENDENCIES, f"{name} is provided by the host"
        assert name not in plugin.PLUGIN_OPTIONAL_DEPENDENCIES, f"{name} is provided by the host"


def test_tags_are_a_short_list():
    assert isinstance(plugin.PLUGIN_TAGS, list)
    assert 1 <= len(plugin.PLUGIN_TAGS) <= 3


def test_supported_version_range_is_declared():
    assert "4" in plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION


def test_module_has_no_run_attribute():
    """A module-level run() makes the host add its own Plugins-menu entry.

    The host does this for any module exposing `run` (ui/plugin_menu_manager.py),
    so keeping one alongside the entry registered in initialize() would show the
    plugin twice.
    """
    assert not hasattr(plugin, "run"), (
        "rename run() to something private, or the plugin appears twice in the menus"
    )


def test_module_has_no_autorun_attribute():
    """autorun() executes at startup; this plugin has nothing to do there."""
    assert not hasattr(plugin, "autorun")


# -- registration ----------------------------------------------------------


def test_initialize_registers_exactly_one_entry_point(context):
    entries = context.menu_actions + context.export_actions + context.analysis_tools
    assert len(entries) == 1, "register one menu entry, or the host shows duplicates"


def test_initialize_registers_session_handlers(context):
    assert len(context.save_handlers) == 1
    assert len(context.load_handlers) == 1
    assert len(context.reset_handlers) == 1


def test_save_handler_is_silent_until_the_dialog_is_used(context):
    """An untouched plugin must not write anything into the user's project."""
    assert context.save_handlers[0]() == {}


def test_save_handler_emits_settings_after_use(context):
    plugin._dialog_opened = True
    plugin.current_settings["repeat"] = 4
    assert context.save_handlers[0]()["settings"]["repeat"] == 4


def test_load_handler_restores_settings(context):
    context.load_handlers[0]({"settings": {"repeat": 7}})
    assert plugin.current_settings["repeat"] == 7


def test_load_handler_ignores_junk(context):
    before = dict(plugin.current_settings)
    context.load_handlers[0](None)
    context.load_handlers[0]({})
    context.load_handlers[0]({"settings": "not a dict"})
    assert plugin.current_settings == before


def test_reset_handler_restores_defaults(context):
    plugin._dialog_opened = True
    plugin.current_settings["repeat"] = 9
    context.reset_handlers[0]()
    assert plugin.current_settings == plugin.get_default_settings()
    assert plugin._dialog_opened is False


def test_reset_handler_leaves_an_open_dialog_alone(context):
    class _Dialog:
        def isVisible(self):
            return True

    context.windows[plugin.WINDOW_ID] = _Dialog()
    plugin.current_settings["repeat"] = 9
    context.reset_handlers[0]()
    assert plugin.current_settings["repeat"] == 9


def test_load_handler_survives_a_deleted_dialog(context):
    class _Dialog:
        def apply_settings(self, settings):
            raise RuntimeError("wrapped C/C++ object of type ... has been deleted")

    context.windows[plugin.WINDOW_ID] = _Dialog()
    context.load_handlers[0]({"settings": {"repeat": 5}})
    assert plugin.current_settings["repeat"] == 5
