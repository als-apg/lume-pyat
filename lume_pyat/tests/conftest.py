"""Shared fixtures: a small, generic, stable test ring.

The ring is a plain FODO lattice with no relationship to any real machine. It
exists so the package can be exercised end to end without shipping lattice
data: eight identical cells, each with a focusing/defocusing quadrupole pair,
a sextupole pair, a horizontal and a vertical corrector, a dipole, and a
monitor. It is stable in both planes with a comfortable margin, so tests may
perturb magnet strengths without losing the closed orbit.
"""

import at
import numpy as np
import pytest

N_CELLS = 8

# Quadrupole gradient chosen for the widest stability margin: it puts the
# horizontal and vertical half-traces near -1.4 and -0.9, well inside +/-2.
QUAD_K = 1.0


def build_test_ring() -> at.Lattice:
    """Build a fresh eight-cell FODO ring.

    Callers get their own lattice: :class:`~lume_pyat.simulator.PyATSimulator`
    mutates the lattice it is given in place, so tests must not share one.
    """
    elements: list[at.Element] = []
    for cell in range(1, N_CELLS + 1):
        tag = f"{cell:02d}"
        elements += [
            at.Monitor(f"BPM_{tag}"),
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QUAD_F_{tag}", 0.3, QUAD_K),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(f"SEXT_F_{tag}", 0.15, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(f"COR_H_{tag}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BEND_{tag}", 1.0, 2 * np.pi / N_CELLS),
            at.Corrector(f"COR_V_{tag}", 0.0, [0.0, 0.0]),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(f"SEXT_D_{tag}", 0.15, -1.0),
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QUAD_D_{tag}", 0.3, -QUAD_K),
            at.Drift("DRIFT", 0.4),
        ]

    ring = at.Lattice(elements, name="TEST_RING", energy=1.0e9, periodicity=1)
    # 4D optics: the ring carries no RF cavity.
    ring.disable_6d()
    return ring


@pytest.fixture
def test_ring() -> at.Lattice:
    """A fresh :func:`build_test_ring` lattice per test."""
    return build_test_ring()
