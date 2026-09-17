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


# Harmonic number of the 6D ring's cavity. Any integer works; this one keeps
# the RF frequency in the hundreds of MHz for a ring this size.
HARMONIC_NUMBER = 32

# Cavity voltage of the 6D ring, in volts. Radiation is off, so any positive
# voltage gives a stable synchrotron motion; this one is a typical order of
# magnitude for a small ring.
RF_VOLTAGE = 1.0e6


def build_test_ring_6d() -> at.Lattice:
    """The same FODO ring with one RF cavity, solved in 6D.

    The cavity is enabled and radiation is left off -- the configuration the
    package supports for 6D rings. Its frequency is set to the ring's
    revolution frequency times :data:`HARMONIC_NUMBER`, so the nominal orbit
    sits on-momentum and is flat, exactly like the 4D ring's.
    """
    ring = build_test_ring()
    frequency = HARMONIC_NUMBER * ring.get_revolution_frequency()
    ring.insert(
        0,
        at.RFCavity("RF", 0.0, RF_VOLTAGE, frequency, HARMONIC_NUMBER, ring.energy),
    )
    ring.enable_6d(at.RFCavity)
    return ring


def strip_monitors(ring: at.Lattice) -> at.Lattice:
    """The same ring with every ``at.Monitor`` removed.

    For the monitorless contract: a ring with no monitors solves to an empty
    reading rather than raising, and this is how the tests build one.
    """
    monitorless = at.Lattice(
        [element for element in ring if not isinstance(element, at.Monitor)],
        name="NO_MONITORS",
        energy=ring.energy,
        periodicity=1,
    )
    # Keep the ring's dimensionality: a 6D ring stays 6D (cavity on, radiation
    # off, as build_test_ring_6d sets it up), a 4D ring stays 4D.
    if ring.is_6d:
        monitorless.enable_6d(at.RFCavity)
    else:
        monitorless.disable_6d()
    return monitorless


@pytest.fixture
def test_ring() -> at.Lattice:
    """A fresh :func:`build_test_ring` lattice per test."""
    return build_test_ring()


@pytest.fixture
def test_ring_6d() -> at.Lattice:
    """A fresh :func:`build_test_ring_6d` lattice per test."""
    return build_test_ring_6d()
