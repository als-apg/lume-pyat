"""PyATSimulator: lookup, validate-then-apply, error folding, last_solution."""

import dataclasses

import at
import numpy as np
import pytest

from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.simulator import (
    ElementState,
    PyATSimulator,
    restore_element,
    snapshot_element,
)
from lume_pyat.tests.conftest import N_CELLS, QUAD_K


@pytest.fixture
def simulator(test_ring):
    return PyATSimulator(test_ring)


def destabilise(ring):
    for index in ring.get_uint32_index("QUAD_F_*"):
        ring[index].K = 3.0


# -- element lookup --------------------------------------------------------


def test_element_index_finds_elements_by_name(simulator, test_ring):
    index = simulator.element_index("QUAD_F_01")
    assert test_ring[index].FamName == "QUAD_F_01"


def test_element_index_rejects_an_unknown_name(simulator):
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        simulator.element_index("NOPE")


def test_element_returns_the_live_element(simulator):
    element = simulator.element("QUAD_F_01")
    element.K = 1.75
    # The same object the lattice holds, not a copy.
    assert simulator.lattice[simulator.element_index("QUAD_F_01")].K == 1.75


def test_element_rejects_an_unknown_name(simulator):
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        simulator.element("NOPE")


# -- lattice ownership -----------------------------------------------------


def test_the_lattice_is_adopted_not_copied(test_ring):
    simulator = PyATSimulator(test_ring)
    assert simulator.lattice is test_ring


def test_writes_are_idempotent_and_order_independent(test_ring):
    simulator = PyATSimulator(test_ring)

    simulator.element("QUAD_F_01").K = 1.2
    simulator.element("QUAD_F_01").K = 1.2
    simulator.element("QUAD_D_01").K = -1.3
    twice_then_other = simulator.solve()

    simulator.element("QUAD_D_01").K = -1.3
    simulator.element("QUAD_F_01").K = 1.2
    other_order = simulator.solve()

    assert twice_then_other == other_order


# -- misalignment seeding --------------------------------------------------


def test_misalignments_are_applied_at_construction(test_ring):
    PyATSimulator(test_ring, element_misalignments={"QUAD_F_01": {"dx": 300e-6}})

    index = test_ring.get_uint32_index("QUAD_F_01")[0]
    assert test_ring[index].T1[0] == pytest.approx(-300e-6)


def test_every_misalignment_name_is_validated_before_any_element_is_mutated(test_ring):
    # The transform is absolute, so a partial application could not be undone
    # by retrying. Ordering here is the whole safety property.
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        PyATSimulator(
            test_ring,
            element_misalignments={"QUAD_F_01": {"dx": 300e-6}, "NOPE": {"dy": 1e-6}},
        )

    index = test_ring.get_uint32_index("QUAD_F_01")[0]
    assert not hasattr(test_ring[index], "T1")


def test_no_misalignments_leaves_the_lattice_untouched(test_ring):
    PyATSimulator(test_ring)
    index = test_ring.get_uint32_index("QUAD_F_01")[0]
    assert not hasattr(test_ring[index], "T1")


def test_seeded_misalignments_survive_later_writes(test_ring):
    simulator = PyATSimulator(
        test_ring, element_misalignments={"QUAD_F_01": {"dy": 300e-6}}
    )
    misaligned = simulator.solve()

    # A write elsewhere must not rebuild the lattice and lose the seed.
    simulator.element("COR_H_03").KickAngle = [1e-5, 0.0]
    simulator.solve()
    simulator.element("COR_H_03").KickAngle = [0.0, 0.0]

    assert simulator.solve() == misaligned


# -- solving ---------------------------------------------------------------


def test_solve_returns_one_reading_per_monitor(simulator):
    solution = simulator.solve()
    assert sorted(solution) == [f"BPM_{cell:02d}" for cell in range(1, N_CELLS + 1)]
    assert all(len(xy) == 2 for xy in solution.values())
    assert all(np.isfinite(v) for xy in solution.values() for v in xy)


def test_solve_reflects_lattice_mutations(simulator):
    assert max(abs(x) for x, _ in simulator.solve().values()) == pytest.approx(0.0)

    simulator.element("COR_H_03").KickAngle = [1e-4, 0.0]
    assert max(abs(x) for x, _ in simulator.solve().values()) > 1e-5


def test_an_unstable_lattice_raises(simulator):
    destabilise(simulator.lattice)
    with pytest.raises(OrbitSolveError, match="one-turn matrix unstable"):
        simulator.solve()


@pytest.mark.parametrize(
    "raised",
    [
        at.AtError("pyAT detected a failure"),
        np.linalg.LinAlgError("singular matrix"),
        ValueError("non-convergence"),
    ],
    ids=["AtError", "LinAlgError", "ValueError"],
)
def test_solver_exceptions_are_folded_into_orbit_solve_error(
    simulator, monkeypatch, raised
):
    def raise_it(*args, **kwargs):
        raise raised

    monkeypatch.setattr("lume_pyat.simulator.solve_orbit", raise_it)

    with pytest.raises(OrbitSolveError, match=type(raised).__name__) as excinfo:
        simulator.solve()
    assert excinfo.value.__cause__ is raised


def test_the_fold_is_scoped_to_the_solve_call(simulator, monkeypatch):
    # UnknownElementError is itself a ValueError, so a fold that reached past
    # the solve call would relabel "unknown element" as "unstable orbit". Only
    # ValueErrors out of solve_orbit are folded; one raised anywhere else in
    # solve() must pass through unchanged.
    def raise_it(*args, **kwargs):
        raise ValueError("not a solve failure")

    monkeypatch.setattr("lume_pyat.simulator.monitor_xy", raise_it)

    with pytest.raises(ValueError, match="not a solve failure") as excinfo:
        simulator.solve()
    assert not isinstance(excinfo.value, OrbitSolveError)


# -- last_solution ---------------------------------------------------------


def test_last_solution_before_any_solve_is_a_clear_error(simulator):
    with pytest.raises(OrbitSolveError, match="no closed orbit has been solved yet"):
        _ = simulator.last_solution


def test_last_solution_matches_the_most_recent_successful_solve(simulator):
    assert simulator.solve() == simulator.last_solution

    simulator.element("COR_H_03").KickAngle = [1e-4, 0.0]
    assert simulator.solve() == simulator.last_solution


def test_a_failed_solve_leaves_the_previous_solution_in_place(simulator):
    good = simulator.solve()

    destabilise(simulator.lattice)
    with pytest.raises(OrbitSolveError):
        simulator.solve()

    assert simulator.last_solution == good


def test_a_first_failed_solve_leaves_last_solution_unset(simulator):
    destabilise(simulator.lattice)
    with pytest.raises(OrbitSolveError):
        simulator.solve()

    with pytest.raises(OrbitSolveError, match="no closed orbit has been solved yet"):
        _ = simulator.last_solution


# -- the element snapshot trio ---------------------------------------------


def test_snapshot_and_restore_round_trip_a_magnet(test_ring):
    element = test_ring[test_ring.get_uint32_index("QUAD_F_01")[0]]
    state = snapshot_element(element)

    element.K = 2.5
    assert element.K == pytest.approx(2.5)

    restore_element(element, state)
    assert element.K == pytest.approx(QUAD_K)


def test_snapshot_and_restore_round_trip_a_corrector(test_ring):
    element = test_ring[test_ring.get_uint32_index("COR_H_03")[0]]
    state = snapshot_element(element)

    element.KickAngle = [1e-4, 2e-4]
    restore_element(element, state)
    assert list(element.KickAngle) == [0.0, 0.0]


def test_a_snapshot_records_only_the_fields_the_element_carries(test_ring):
    quad = snapshot_element(test_ring[test_ring.get_uint32_index("QUAD_F_01")[0]])
    corrector = snapshot_element(test_ring[test_ring.get_uint32_index("COR_H_03")[0]])

    assert quad.polynom_b is not None and quad.kick_angle is None
    assert corrector.kick_angle is not None and corrector.polynom_b is None


def test_a_snapshot_is_a_copy_not_a_view(test_ring):
    element = test_ring[test_ring.get_uint32_index("QUAD_F_01")[0]]
    state = snapshot_element(element)

    element.PolynomB[1] = 9.9
    assert state.polynom_b[1] == pytest.approx(QUAD_K)


def test_element_state_is_immutable():
    state = ElementState(polynom_b=[0.0, 1.0], kick_angle=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.polynom_b = [1.0]
