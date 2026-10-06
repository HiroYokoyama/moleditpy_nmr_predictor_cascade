# MoleditPy Plugin Template

[![MoleditPy](https://img.shields.io/badge/MoleditPy->=4.0.0-3577F7)](https://github.com/HiroYokoyama/python_molecular_editor)

A starting point for a **third-party [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor)
plugin**: a working example plugin, a headless test suite, an API compatibility
check against the host, and a tag-driven release workflow.

Click **Use this template** on GitHub, then work through the checklist below.

> **Read the [Plugin Development Manual (V4)](https://github.com/HiroYokoyama/python_molecular_editor/blob/main/docs/PLUGIN_DEVELOPMENT_MANUAL_V4.md) first**
> ([rendered version](https://hiroyokoyama.github.io/python_molecular_editor/docs/PLUGIN_DEVELOPMENT_MANUAL_V4.html)).
> It is the reference for the whole `PluginContext` API — every menu, export,
> analysis, file-opener and 3D-style hook, session persistence, and the thread
> rules. This template only demonstrates a small slice of it, and it cannot tell
> you which hook suits your plugin.
>
> Worth reading before you start: §7 on the legacy `run()` / `autorun()` entry
> points (which is why this template opens its dialog through a private
> function — see step 3), and §8.1 on thread safety, since every callback runs on
> the UI thread and a slow one freezes the whole application.

## The easiest possible plugin

You do not need this repository — or any repository — to write a plugin. Save
this as `atom_counter.py` in your plugins directory
(`%USERPROFILE%\.moleditpy\plugins` on Windows, `~/.moleditpy/plugins` on
macOS/Linux) and restart MoleditPy:

```python
PLUGIN_NAME = "Atom Counter"
PLUGIN_VERSION = "1.0.0"

def run(main_window):
    mol = main_window.current_mol
    if mol is None:
        main_window.statusBar().showMessage("No molecule loaded.", 3000)
        return
    main_window.statusBar().showMessage(f"Atoms: {mol.GetNumAtoms()}", 4000)
```

That is the whole plugin. The host sees `run()` and adds **Plugins → Atom
Counter** for you, so there is no menu registration to write. It is the right
shape for a one-action tool you only use yourself.

Reach for this template when you want any of what that approach gives up: the
stable `PluginContext` proxy instead of raw `main_window`, settings that survive
a project reload, tests, a check against the host API, and a release other people
can install. The rest of this README covers those.

## What you get

| Path | What it is |
|---|---|
| `my_plugin/__init__.py` | Metadata the registry reads, `initialize(context)`, and the three session handlers |
| `my_plugin/main_dialog.py` | A small dialog wired the way the host expects |
| `tests/test_plugin.py` | Metadata + registration tests (keep these — they catch the usual mistakes) |
| `tests/test_dialog.py` | Real-PyQt6 dialog tests, offscreen |
| `tests/test_api.py` + `plugin_api_checker.py` | Fails if you touch a host attribute that does not exist |
| `.github/workflows/test.yml` | CI on Linux + Windows, Python 3.11 and 3.13 |
| `.github/workflows/release.yml` | `v*.*.*` tag → zip → GitHub release |
| `.moleditpy-api-allowlist` | Escape hatch for host attributes the checker cannot see |
| `my_single_file_plugin.py` | The same plugin as a single file, at the root where that shape belongs |
| `examples/single_file/` | The two workflow files a single-file plugin needs |

## Single file or package?

This template is a **package** (`my_plugin/` with an `__init__.py`), which is what
you want once a plugin outgrows one screen of code: separate modules, importable
helpers, and tests that mirror them.

A **single `.py` file** is equally valid and is what most plugins in the registry
are — MoleditPy loads a lone script dropped into the plugins directory exactly as
it loads a package folder. Choose it for a one-action tool with no helper modules.

**`my_single_file_plugin.py` at the root is a complete working one** — that is
where the shape belongs, since the release workflow attaches the script itself.
To use it: rename it, copy `examples/single_file/release.yml` and `test.yml` over
the ones in `.github/workflows/` (see
[the notes there](examples/single_file/README.md)), then delete `my_plugin/` and
`examples/`.

The example is covered by this repository's own tests, so it stays working.
63 of the registry's 83 entries are single files today, including
[`3d_molecule_on_2d`](https://github.com/HiroYokoyama/moleditpy_3d-molecule-on-2d),
whose release workflow the example's is modelled on.

Two differences worth knowing: a loose script has nowhere to keep a
`settings.json` beside itself the way `SETTINGS_FILE` does here, so persist state
through the save/load handlers; and keep the Qt import inside the function that
builds the dialog, so importing the module never needs a GUI toolkit.

<details>
<summary>The workflow change, if you would rather edit yours by hand</summary>

   ```yaml
   - name: Create GitHub Release
     env:
       GH_TOKEN: ${{ github.token }}
     run: |
       gh release create "v$VERSION" \
         --title "My Plugin $VERSION" \
         --generate-notes \
         "my_plugin.py"
   ```

   In `.github/workflows/test.yml`, change `--cov=my_plugin` to
   `--cov=my_plugin.py`.

</details>

Registration accepts either form: the asset URL may end in `.py` or `.zip`, and
the metadata constants are read the same way (for a zip, from the package's
`__init__.py`).

## Getting started

1. **Rename the package.** `my_plugin/` → your plugin's name (snake_case). Update
   the same name in both workflow files and in the test imports.
2. **Edit the metadata** at the top of `my_plugin/__init__.py`. `PLUGIN_VERSION`
   must be semver — the release workflow refuses a tag that does not match it.
3. **Pick one entry point.** The host adds its own **Plugins** menu entry for any
   module that exposes `run()`, using `PLUGIN_NAME` — so a plugin that defines
   both `run()` and `add_menu_action()` appears **twice**. This template opens its
   dialog through a private `_open_dialog()` and registers the entry once in
   `initialize()`; `tests/test_plugin.py` fails if a module-level `run` reappears.
   If you want the automatic Plugins entry instead, name it `run()` and drop the
   `add_menu_action()` call.
4. **Write your plugin** against the
   [Plugin Development Manual](https://github.com/HiroYokoyama/python_molecular_editor/blob/main/docs/PLUGIN_DEVELOPMENT_MANUAL_V4.md),
   keeping `PLUGIN_DEPENDENCIES` to packages the host does *not* already ship
   (PyQt6, rdkit and numpy are the host's), and putting anything behind a
   guarded import in `PLUGIN_OPTIONAL_DEPENDENCIES` instead. Move anything slow off the UI thread
   (manual §8.1) — a blocking callback freezes the whole application.
5. **Delete what you are not using.** The template ships both shapes and both
   sets of examples, so a copy always has spare parts:

   | Building a package | Building a single file |
   |---|---|
   | delete `examples/` | delete `my_plugin/` |
   | delete `my_single_file_plugin.py` and `tests/test_single_file_example.py` | copy the two workflows from `examples/single_file/` over `.github/workflows/`, then delete `my_plugin/`, `examples/` and `tests/test_dialog.py` |

   Also strip the parts of `README.md` that describe the shape you dropped, and
   replace this file's content with your own — a released plugin's README is what
   users see in the Plugin Manager.
6. **Run the tests** (below) and keep them green.

```bash
python -m pytest tests/ -v
```

To make the API check actually run, clone the main app next to your repo:

```bash
git clone --depth 1 https://github.com/HiroYokoyama/python_molecular_editor.git ..
```

Without it that test **skips** rather than fails, which is easy to miss — CI
clones it for you.

## Releasing

```bash
git tag v0.1.0        # must equal PLUGIN_VERSION
git push origin v0.1.0
```

The workflow verifies the tag against `PLUGIN_VERSION`, zips the package
(README and LICENSE included) and creates the GitHub release with
`my_plugin_0.1.0.zip` attached. Users can install that zip directly through the
MoleditPy Plugin Manager.

Watch CI go green **before** tagging: a tag cannot be un-released cleanly.

## Getting into the plugin registry

Listing your plugin in the
[registry](https://github.com/HiroYokoyama/moleditpy-plugins) is a **separate,
manual step, and it is not automated for third-party repos**. The registry
updates itself from a repository dispatch that needs a `REGISTRY_PAT` secret,
which only the registry maintainer can issue.

The release workflow tries the dispatch **only if that secret exists** on your
repository: maintainers get automatic registration, everyone else gets a no-op
plus a log line pointing at the manual route. Nothing fails either way.

So, after your release is published:

1. Open a **Request Plugin Registration / Update** issue on the registry repo:
   <https://github.com/HiroYokoyama/moleditpy-plugins/issues/new/choose>
2. Fill in the release asset URL, e.g.
   `https://github.com/<you>/<repo>/releases/download/v0.1.0/my_plugin_0.1.0.zip`,
   and the asset's SHA-256 (the form asks for it, and registration verifies it):
   - PowerShell: `Get-FileHash ./my_plugin_0.1.0.zip -Algorithm SHA256`
   - bash: `shasum -a 256 my_plugin_0.1.0.zip`
3. A maintainer runs the registration script against that URL, which reads your
   `PLUGIN_*` constants and records the hash.

For later versions, open an issue again with the new release URL (or ask to be
granted the dispatch secret if you release often).

## Conventions worth keeping

- **Do not write to the user's project until your dialog is actually used** —
  the save handler returns `{}` until then.
- **Guard every host call.** Wrap `context.*` access that may not exist in older
  hosts with `hasattr`, and add it to `.moleditpy-api-allowlist` with a reason.
- **Never block on a modal dialog during construction** — it deadlocks headless
  CI and startup.
- **Write files with `newline="\n"`** so output is identical on every platform.
- **Ship no secrets.** Nothing in this template needs a token; the release
  workflow uses only the automatic `GITHUB_TOKEN`.

## Licence

This template is GPL-3.0 (see [LICENSE](LICENSE)), matching the MoleditPy
ecosystem. Replace it if your plugin uses a different licence.
