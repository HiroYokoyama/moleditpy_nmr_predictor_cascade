"""Independent weak-coupling references; no predicted shifts or fitted J values."""

import importlib.util
import itertools
import math
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = next(_ROOT.glob("*/coupling.py"))
_SPEC = importlib.util.spec_from_file_location("reference_coupling", _SOURCE)
coupling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coupling)


@pytest.mark.parametrize("frequency", [100.0, 400.0, 800.0])
@pytest.mark.parametrize(
    "partners", [(7,), (7, 7, 7), (12, 3), (12, 3, 3), (7, 7, 7, 7, 7, 7)]
)
def test_first_order_spin_state_populations(partners, frequency):
    # Secular Hamiltonian: observed frequency offset = sum(J_k m_k),
    # with independent m_k = +/-1/2 and equally populated partner states.
    states = list(itertools.product((-0.5, 0.5), repeat=len(partners)))
    populations = Counter(
        sum(j * m for j, m in zip(partners, state)) for state in states
    )
    pattern = list(Counter(partners).items())
    lines = coupling.multiplet_lines(3.25, pattern, frequency)
    expected = [
        (3.25 + offset / frequency, count / len(states))
        for offset, count in sorted(populations.items())
    ]
    assert len(lines) == len(expected)
    for actual, reference in zip(lines, expected):
        assert actual == pytest.approx(reference, abs=1e-12)
    assert sum(height for _, height in lines) == pytest.approx(1.0, abs=1e-12)
    assert sum(ppm * height for ppm, height in lines) == pytest.approx(3.25, abs=1e-12)


@pytest.mark.parametrize("frequency", [100.0, 400.0, 800.0])
def test_lorentzian_half_height_and_finite_window_integral(frequency):
    # Peak-normalized line, not unit-area density: integral[-a,a]
    # = 2 H gamma atan(a/gamma), gamma=FWHM/2 in ppm.
    height, width_hz = 3.0, 2.0
    gamma = width_hz / (2 * frequency)
    extent = 20 * gamma
    xs, ys = coupling.lorentzian_curve(
        [(0.0, height)], width_hz, frequency, -extent, extent, grid_points=20001
    )
    assert ys[xs.index(0.0)] == pytest.approx(height, abs=1e-12)
    for sign in (-1, 1):
        index = min(range(len(xs)), key=lambda i: abs(xs[i] - sign * gamma))
        assert ys[index] == pytest.approx(height / 2, abs=1e-12)
    area = sum((b - a) * (ya + yb) / 2 for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]))
    assert area == pytest.approx(2 * height * gamma * math.atan(20), rel=1e-7)


def test_deuterium_does_not_contribute_to_proton_coupled_carbon_multiplet():
    chem = pytest.importorskip("rdkit.Chem")
    mol = chem.AddHs(chem.MolFromSmiles("[2H]C([2H])([2H])C"))
    carbons = coupling.predict_ch_couplings(mol)
    assert sorted(item["n_h"] for item in carbons.values()) == [0, 3]
    assert sorted(item["mult"] for item in carbons.values()) == ["q", "s"]
