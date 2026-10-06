"""NMR Predictor (CASCADE) — 1H / 13C shift prediction with coupling estimates.

Shifts come from the CASCADE web server of the Paton group (Colorado State
University): the structure is sent there as a SMILES string. Couplings and
multiplicities are estimated locally by ``coupling.py``.
"""

import json
import logging
import os

# --------------------------------------------------------------------------
# Registry metadata — read by the plugin registry scripts; keep the names.
# --------------------------------------------------------------------------

PLUGIN_NAME = "NMR Predictor (CASCADE)"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Predict 1H and 13C NMR shifts with CASCADE, with estimated J couplings and "
    "multiplets. Sends the structure (as SMILES) to the external CASCADE web "
    "server (nova.chem.colostate.edu); requires an internet connection."
)
PLUGIN_CATEGORY = "Analysis"
PLUGIN_TAGS = ["Analysis"]
PLUGIN_DEPENDENCIES = ["matplotlib"]  # PyQt6/rdkit/numpy/pyvista are the host's
PLUGIN_OPTIONAL_DEPENDENCIES = []
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"

MENU_PATH = "Analysis/NMR Prediction (CASCADE)..."
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
RESULT_WINDOW_ID = "cascade_nmr_result"

_context = None


def get_default_settings():
    from .cascade_client import DEFAULT_SERVER

    return {
        "nucleus": "1H",
        "spectrometer_mhz": 400.0,  # 1H frequency; 13C uses a quarter of it
        "show_multiplets": True,
        "symmetrize": True,
        "server": DEFAULT_SERVER,
    }


def load_settings(path=None):
    """Defaults overlaid with settings.json (a missing or broken file is ignored)."""
    settings = get_default_settings()
    path = path or SETTINGS_FILE
    try:
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except FileNotFoundError:
        return settings
    except (OSError, ValueError) as exc:
        logging.warning("%s: ignoring unreadable %s: %s", PLUGIN_NAME, path, exc)
        return settings
    if isinstance(saved, dict):
        for key, default in settings.items():
            if key in saved and isinstance(saved[key], type(default)):
                settings[key] = saved[key]
            elif key in saved and isinstance(default, float) and isinstance(saved[key], int):
                settings[key] = float(saved[key])
    return settings


def save_settings(settings, path=None):
    path = path or SETTINGS_FILE
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(settings, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        logging.warning("%s: could not write %s: %s", PLUGIN_NAME, path, exc)


def _open_prediction(context):
    """Menu callback. Deliberately not named ``run`` — the host would add a
    second Plugins-menu entry for a module-level ``run()``."""
    from .ui import start_prediction

    start_prediction(context)


def initialize(context):
    """Entry point for the V4 plugin API."""
    global _context
    _context = context
    context.add_menu_action(MENU_PATH, lambda: _open_prediction(context))
