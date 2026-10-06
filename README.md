# NMR Predictor (CASCADE) for MoleditPy

[![CI](https://github.com/HiroYokoyama/moleditpy_nmr_predictor_cascade/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_nmr_predictor_cascade/actions/workflows/test.yml)
[![MoleditPy](https://img.shields.io/badge/MoleditPy->=4.0.0-3577F7)](https://github.com/HiroYokoyama/python_molecular_editor)
[![GitHub tag](https://img.shields.io/github/v/tag/HiroYokoyama/moleditpy_nmr_predictor_cascade?label=version)](https://github.com/HiroYokoyama/moleditpy_nmr_predictor_cascade/tags)
[![GitHub Downloads](https://img.shields.io/github/downloads/HiroYokoyama/moleditpy_nmr_predictor_cascade/total)](https://github.com/HiroYokoyama/moleditpy_nmr_predictor_cascade/releases)

A [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor) plugin
that predicts **1H** and **13C** NMR chemical shifts with
[CASCADE](https://nova.chem.colostate.edu/v2/cascade/), and estimates the
**coupling pattern** of every signal (s, d, t, q, dd, dq, ... with J in Hz).

![1H result for methyl (E)-crotonate](img/main.png)

*MoleditPy with methyl (E)-crotonate, 1H shifts as returned by CASCADE; the selected olefinic
proton is a dq (J = 17.0, 7.0 Hz).*

> **This plugin sends your structure to an external server.**
> The molecule is sent as a SMILES string to the CASCADE web server run by the
> Paton group at Colorado State University (`nova.chem.colostate.edu`). Every
> prediction shows the exact SMILES and asks before anything is sent. Do not
> use it for structures you must keep confidential. An internet connection is
> required.

## Features

* **1H and 13C shifts** from CASCADE, a graph neural network evaluated on an
  MMFF conformer ensemble (Boltzmann-weighted on the server), with a
  per-atom confidence (High / Moderate / Low).
* **Coupling estimates** (made locally, see below):
  * 1H: multiplicity and J for every proton; the spectrum can be drawn with
    the multiplets split at your spectrometer frequency.
  * 13C: the carbon type (C / CH / CH2 / CH3), the 1H-coupled multiplicity
    (s / d / t / q) and 1J(CH).
* **Line broadening** (on by default): Lorentzian lines instead of sticks,
  1 Hz wide for 1H and 2 Hz for 13C, adjustable or switched off.
* **Interactive spectrum** (matplotlib): hover or click a peak to highlight
  its atoms in the 3D view; picking an atom in 3D selects its row.
* Symmetry-equivalent atoms are averaged into one signal (optional).
* **CSV export** of the table.

## Usage

1. Draw or load a molecule (3D is recommended: stereochemistry is read from
   the 3D structure).
2. **Analysis > NMR Prediction (CASCADE)...**
3. Pick the nucleus and spectrometer frequency, check the SMILES that will be
   sent, and press **Send & Predict**. A job usually takes a few seconds to a
   minute; it can be cancelled.

   ![The confirmation shown before anything is sent](img/confirm.png)

4. In the result window, hover or click peaks (or table rows) to highlight the
   atoms in 3D. **Graph Controls** set the ppm range, the multiplets and the
   line broadening; **Export CSV** saves the table.

The choices are remembered in `settings.json` next to the plugin.

![13C result for toluene](img/13c.png)

*Toluene, 13C: carbon types and 1J(CH) next to the CASCADE shifts.*

## How the couplings are estimated

CASCADE predicts shifts only. The multiplets come from `coupling.py`, a small
rule-based module shared with the
[nmrshiftdb2 predictor](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2):

| Coupling | J (Hz) |
|---|---|
| 2J geminal, sp3 / =CH2 | 12 / 2 |
| 3J across a freely rotating single bond | 7 |
| 3J across a ring bond | Karplus curve on the 3D dihedral |
| 3J alkene cis / trans | 10 / 17 |
| 3J aromatic ortho, 4J meta | 7.5 / 1.5 |
| 3J aldehyde H-C(=O)-C-H | 2.5 |
| 1J(CH) sp3 / sp2 / aromatic / sp / formyl | 125 / 157 / 159 / 249 / 172, plus heteroatom increments |

OH, NH and SH protons are treated as exchanging (uncoupled). Spectra are drawn
first-order. These are typical values for orientation, not a fit to a real
spectrum.

![Olefinic region at 400 MHz with 1 Hz lines](img/multiplets.png)

*The olefinic region of methyl (E)-crotonate at 400 MHz: the dq at 7.19 ppm and
the d at 5.88 ppm.*

The About box shows the version of the shared module
(`COUPLING_MODULE_VERSION`), so the two plugins can be checked for being in
step.

## Requirements

* MoleditPy 4.x (provides PyQt6, RDKit, PyVista)
* `matplotlib`
* Internet access to `nova.chem.colostate.edu`

## Credits and citation

CASCADE is developed by the Paton group, Colorado State University. If you use
its predictions, please cite:

* Y. Guan, S. V. Shree Sowndarya, L. C. Gallegos, P. C. St. John, R. S. Paton,
  *Real-time prediction of 1H and 13C chemical shifts with DFT accuracy using a
  3D graph neural network*, Chem. Sci. **2021**, 12, 12012-12026.

The web service is provided by its authors free of charge; this plugin is not
affiliated with them. Please use it considerately.

## Licence

GPL-3.0 (see [LICENSE](LICENSE)).
