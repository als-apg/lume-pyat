"""The three solve guards, monitor selection, and warning suppression."""

import warnings

import at
import numpy as np
import pytest

from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.solve import _monitor_refpts, monitor_xy, solve_orbit
from lume_pyat.tests.conftest import N_CELLS, build_test_ring


def destabilise(ring: at.Lattice) -> at.Lattice:
    """Over-focus every focusing quadrupole until the one-turn map is unstable."""
    for index in ring.get_uint32_index("QUAD_F_*"):
        ring[index].K = 3.0
    return ring


def test_solve_returns_a_finite_orbit_at_every_monitor(test_ring):
    orbit = solve_orbit(test_ring)
    assert orbit.shape == (N_CELLS, 6)
    assert np.all(np.isfinite(orbit))


def test_a_corrector_kick_moves_the_orbit(test_ring):
    assert np.abs(solve_orbit(test_ring)[:, 0]).max() == pytest.approx(0.0, abs=1e-12)

    index = test_ring.get_uint32_index("COR_H_03")[0]
    test_ring[index].KickAngle = [1.0e-4, 0.0]
    assert np.abs(solve_orbit(test_ring)[:, 0]).max() > 1.0e-5


def test_monitors_are_selected_by_type_not_by_name(test_ring):
    # Row -> element alignment is what lets a caller pair readings with
    # elements, so selection must not depend on a naming convention.
    index = test_ring.get_uint32_index("BPM_04")[0]
    test_ring[index].FamName = "SOMETHING_ELSE"

    refpts = _monitor_refpts(test_ring)
    assert len(refpts) == N_CELLS
    assert index in refpts
    assert [test_ring[i].FamName for i in refpts][3] == "SOMETHING_ELSE"


def test_monitor_xy_rows_follow_ring_order(test_ring):
    readings = monitor_xy(test_ring, solve_orbit(test_ring))
    assert [name for name, _, _ in readings] == [
        f"BPM_{cell:02d}" for cell in range(1, N_CELLS + 1)
    ]


def test_monitor_xy_reads_columns_zero_and_two(test_ring):
    orbit = np.zeros((N_CELLS, 6))
    orbit[2, 0] = 1.5e-6  # x
    orbit[2, 1] = 9.9  # px — must be ignored
    orbit[2, 2] = -3.5e-6  # y

    readings = monitor_xy(test_ring, orbit)
    assert readings[2] == ("BPM_03", 1.5e-6, -3.5e-6)


def test_non_finite_one_turn_matrix_trips_the_first_guard(test_ring):
    index = test_ring.get_uint32_index("QUAD_F_01")[0]
    test_ring[index].PolynomB[1] = np.nan

    with pytest.raises(OrbitSolveError, match="non-finite entries"):
        solve_orbit(test_ring)


def test_unstable_one_turn_matrix_trips_the_trace_guard(test_ring):
    with pytest.raises(OrbitSolveError, match="one-turn matrix unstable") as excinfo:
        solve_orbit(destabilise(test_ring))

    # The message reports both planes and the threshold it compared against.
    assert "|trace_x|" in str(excinfo.value)
    assert "|trace_y|" in str(excinfo.value)
    assert "threshold 2.0" in str(excinfo.value)


def test_non_finite_closed_orbit_trips_the_last_guard(test_ring, monkeypatch):
    # This guard is defense in depth and cannot be provoked through the
    # lattice alone: at.find_m44 solves the closed orbit internally, so any
    # lattice that yields a non-finite orbit trips the first guard instead.
    # Patching find_orbit4 is the only way to exercise it.
    def non_finite_orbit(ring, refpts=None, **kwargs):
        return np.zeros(6), np.full((N_CELLS, 6), np.nan)

    monkeypatch.setattr(at, "find_orbit4", non_finite_orbit)

    with pytest.raises(OrbitSolveError, match="non-finite closed orbit"):
        solve_orbit(test_ring)


def test_solver_warnings_do_not_reach_the_caller(test_ring, monkeypatch):
    # pyAT signals instability with an AtWarning alongside a garbage return
    # value. solve_orbit suppresses those so a guard trip surfaces as one
    # exception rather than console noise plus an exception -- and so a caller
    # who turned warnings into errors is not derailed by expected noise.
    #
    # The warning is injected rather than provoked: this pyAT version does not
    # actually warn for any lattice state reachable on the test ring, so the
    # suppression blocks would otherwise go unexercised.
    def warn_then_delegate(real):
        def wrapper(*args, **kwargs):
            warnings.warn("expected instability noise", at.AtWarning, stacklevel=2)
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(at, "find_m44", warn_then_delegate(at.find_m44))
    monkeypatch.setattr(at, "find_orbit4", warn_then_delegate(at.find_orbit4))

    with warnings.catch_warnings():
        warnings.simplefilter("error", at.AtWarning)
        orbit = solve_orbit(test_ring)

    assert np.all(np.isfinite(orbit))


def test_a_guard_trip_raises_rather_than_warning(test_ring):
    with warnings.catch_warnings():
        warnings.simplefilter("error", at.AtWarning)
        with pytest.raises(OrbitSolveError):
            solve_orbit(destabilise(test_ring))


def test_a_monitorless_ring_yields_an_empty_result_rather_than_raising():
    # The documented contract. A ring with no monitors is not necessarily a
    # mistake -- driving magnets and reading setpoints back needs none -- so
    # the guards still run and the readout is simply empty. A caller that does
    # expect readings learns so precisely, at model construction -- see the
    # non-monitor binding test in test_model.py.
    ring = build_test_ring()
    monitorless = at.Lattice(
        [element for element in ring if not isinstance(element, at.Monitor)],
        name="NO_MONITORS",
        energy=ring.energy,
        periodicity=1,
    )
    monitorless.disable_6d()

    assert len(_monitor_refpts(monitorless)) == 0
    assert solve_orbit(monitorless).shape == (0, 6)
    assert monitor_xy(monitorless, solve_orbit(monitorless)) == []


def test_a_monitorless_ring_still_trips_the_stability_guards():
    # Empty readout is not a bypass: the one-turn matrix is checked either way.
    ring = destabilise(build_test_ring())
    monitorless = at.Lattice(
        [element for element in ring if not isinstance(element, at.Monitor)],
        name="NO_MONITORS",
        energy=ring.energy,
        periodicity=1,
    )
    monitorless.disable_6d()

    with pytest.raises(OrbitSolveError, match="one-turn matrix unstable"):
        solve_orbit(monitorless)


def test_orbit_solve_error_is_the_canonical_class():
    # solve re-exports it for convenience; exceptions.py stays its home.
    import lume_pyat.exceptions
    import lume_pyat.solve

    assert lume_pyat.solve.OrbitSolveError is lume_pyat.exceptions.OrbitSolveError
    assert lume_pyat.OrbitSolveError is OrbitSolveError
