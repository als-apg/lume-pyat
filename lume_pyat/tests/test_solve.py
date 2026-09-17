"""The three solve guards, the 4D/6D dispatch, monitor selection, warnings."""

import warnings

import at
import numpy as np
import pytest

from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.solve import _monitor_refpts, monitor_xy, solve_orbit
from lume_pyat.tests.conftest import (
    N_CELLS,
    build_test_ring,
    build_test_ring_6d,
    strip_monitors,
)


def destabilise(ring: at.Lattice) -> at.Lattice:
    """Over-focus every focusing quadrupole until the vertical plane is unstable."""
    for index in ring.get_uint32_index("QUAD_F_*"):
        ring[index].K = 3.0
    return ring


def destabilise_both_planes(ring: at.Lattice) -> at.Lattice:
    """Flip every focusing quadrupole; both transverse planes go unstable."""
    for index in ring.get_uint32_index("QUAD_F_*"):
        ring[index].K = -3.0
    return ring


# -- the solve -------------------------------------------------------------


def test_solve_returns_a_finite_orbit_at_every_monitor(test_ring):
    orbit = solve_orbit(test_ring)
    assert orbit.shape == (N_CELLS, 6)
    assert np.all(np.isfinite(orbit))


def test_a_6d_ring_solves_at_every_monitor(test_ring_6d):
    orbit = solve_orbit(test_ring_6d)
    assert orbit.shape == (N_CELLS, 6)
    assert np.all(np.isfinite(orbit))
    # On-momentum and flat: the cavity sits at the ring's revolution harmonic.
    assert np.abs(orbit[:, 0]).max() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(orbit[:, 4]).max() == pytest.approx(0.0, abs=1e-9)


def test_a_6d_ring_is_solved_in_6d(test_ring_6d):
    # Detuning the RF shifts the momentum, and the dispersion turns that into
    # a horizontal orbit. A 4D solve cannot see the cavity at all, so a
    # dispersive orbit here is the proof that the ring was solved in 6D.
    cavity = test_ring_6d[test_ring_6d.get_uint32_index(at.RFCavity)[0]]
    cavity.Frequency *= 1.0 + 1.0e-5

    orbit = solve_orbit(test_ring_6d)
    assert np.abs(orbit[:, 4]).max() > 1.0e-5  # off-momentum ...
    assert np.abs(orbit[:, 0]).max() > 1.0e-6  # ... and therefore displaced


def test_a_4d_ring_is_solved_in_4d(test_ring):
    # pyAT refuses to solve a cavity-less ring in 6D, so the dispatch has to
    # go the other way too: a 4D ring solves, and solves without a cavity.
    assert not test_ring.is_6d
    assert solve_orbit(test_ring).shape == (N_CELLS, 6)


def test_a_corrector_kick_moves_the_orbit(test_ring):
    assert np.abs(solve_orbit(test_ring)[:, 0]).max() == pytest.approx(0.0, abs=1e-12)

    index = test_ring.get_uint32_index("COR_H_03")[0]
    test_ring[index].KickAngle = [1.0e-4, 0.0]
    assert np.abs(solve_orbit(test_ring)[:, 0]).max() > 1.0e-5


# -- monitors --------------------------------------------------------------


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


# -- the guards ------------------------------------------------------------


def test_non_finite_one_turn_matrix_trips_the_first_guard(test_ring):
    index = test_ring.get_uint32_index("QUAD_F_01")[0]
    test_ring[index].PolynomB[1] = np.nan

    with pytest.raises(OrbitSolveError, match="non-finite entries"):
        solve_orbit(test_ring)


def test_an_unstable_ring_trips_the_eigenvalue_guard_naming_the_plane(test_ring):
    with pytest.raises(
        OrbitSolveError, match="one-turn map unstable in the vertical plane"
    ) as excinfo:
        solve_orbit(destabilise(test_ring))

    # The message reports every plane and the limit it compared against.
    message = str(excinfo.value)
    assert "horizontal" in message
    assert "vertical" in message
    assert "longitudinal" in message
    assert "limit 1 + 1e-06" in message


def test_the_eigenvalue_guard_names_every_unstable_plane(test_ring):
    with pytest.raises(
        OrbitSolveError, match="unstable in the horizontal and vertical planes"
    ):
        solve_orbit(destabilise_both_planes(test_ring))


def test_the_eigenvalue_guard_catches_a_finite_but_growing_6d_map(test_ring_6d):
    # The case the 2x2 trace test was replaced for: with a cavity in the ring
    # the one-turn matrix stays finite while an eigenvalue leaves the unit
    # circle, and the guard has to catch it from the full 6x6 spectrum.
    with pytest.raises(
        OrbitSolveError, match="one-turn map unstable in the vertical plane"
    ):
        solve_orbit(destabilise(test_ring_6d))


@pytest.mark.parametrize(
    ("excess", "trips"),
    [(5.0e-7, False), (2.0e-6, True)],
    ids=["within tolerance", "beyond tolerance"],
)
def test_the_eigenvalue_limit_is_one_plus_1e6(test_ring, monkeypatch, excess, trips):
    # The contract: |eigenvalue| <= 1 + 1e-6. Round-off on a stable ring must
    # not trip it; a genuine excursion beyond it must.
    def scaled_identity(ring, *args, **kwargs):
        return np.eye(6) * (1.0 + excess), np.empty((0, 6, 6))

    monkeypatch.setattr(at, "find_m66", scaled_identity)

    if trips:
        with pytest.raises(OrbitSolveError, match="one-turn map unstable"):
            solve_orbit(test_ring)
    else:
        assert solve_orbit(test_ring).shape == (N_CELLS, 6)


def test_non_finite_closed_orbit_trips_the_last_guard(test_ring, monkeypatch):
    # This guard is defense in depth and cannot be provoked through the
    # lattice alone: at.find_m66 solves the closed orbit internally, so any
    # lattice that yields a non-finite orbit trips the first guard instead.
    # Patching the orbit solver is the only way to exercise it.
    def non_finite_orbit(ring, refpts=None, **kwargs):
        return np.zeros(6), np.full((N_CELLS, 6), np.nan)

    monkeypatch.setattr(at, "find_orbit4", non_finite_orbit)

    with pytest.raises(OrbitSolveError, match="non-finite closed orbit"):
        solve_orbit(test_ring)


def test_non_finite_6d_closed_orbit_trips_the_last_guard(test_ring_6d, monkeypatch):
    def non_finite_orbit(ring, refpts=None, **kwargs):
        return np.zeros(6), np.full((N_CELLS, 6), np.nan)

    monkeypatch.setattr(at, "find_orbit6", non_finite_orbit)

    with pytest.raises(OrbitSolveError, match="non-finite closed orbit"):
        solve_orbit(test_ring_6d)


# -- warnings --------------------------------------------------------------


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

    monkeypatch.setattr(at, "find_m66", warn_then_delegate(at.find_m66))
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


# -- monitorless rings -----------------------------------------------------


def test_a_monitorless_ring_yields_an_empty_result_rather_than_raising():
    # The documented contract. A ring with no monitors is not necessarily a
    # mistake -- driving magnets and reading setpoints back needs none -- so
    # the guards still run and the readout is simply empty. A caller that does
    # expect readings learns so precisely, at model construction -- see the
    # non-monitor binding test in test_model.py.
    monitorless = strip_monitors(build_test_ring())

    assert len(_monitor_refpts(monitorless)) == 0
    assert solve_orbit(monitorless).shape == (0, 6)
    assert monitor_xy(monitorless, solve_orbit(monitorless)) == []


def test_a_monitorless_ring_still_trips_the_stability_guards():
    # Empty readout is not a bypass: the one-turn matrix is checked either way.
    monitorless = strip_monitors(destabilise(build_test_ring()))

    with pytest.raises(OrbitSolveError, match="one-turn map unstable"):
        solve_orbit(monitorless)


def test_a_monitorless_6d_ring_keeps_its_cavity():
    # strip_monitors must not strip the cavity along with the monitors, or a
    # 6D ring would silently become a 4D one.
    monitorless = strip_monitors(build_test_ring_6d())

    assert monitorless.is_6d
    assert solve_orbit(monitorless).shape == (0, 6)


# -- the exception ---------------------------------------------------------


def test_orbit_solve_error_is_the_canonical_class():
    # solve re-exports it for convenience; exceptions.py stays its home.
    import lume_pyat.exceptions
    import lume_pyat.solve

    assert lume_pyat.solve.OrbitSolveError is lume_pyat.exceptions.OrbitSolveError
    assert lume_pyat.OrbitSolveError is OrbitSolveError
