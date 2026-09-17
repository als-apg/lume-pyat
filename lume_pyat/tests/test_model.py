"""LUMEPyATModel: construction, the frozen variable set, and atomic writes."""

from typing import ClassVar

import at
import pytest
from lume.actions import WritableActionMixin
from lume.exceptions import ReadOnlyError
from lume.variables import ScalarVariable

from lume_pyat.actions import (
    ElementBinding,
    PyATLatticeScalarVariable,
    PyATReadOnlyScalarVariable,
    PyATWritableScalarVariable,
)
from lume_pyat.exceptions import (
    AmbiguousElementError,
    OrbitSolveError,
    UnknownElementError,
)
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator
from lume_pyat.tests.conftest import QUAD_K, build_test_ring, strip_monitors

UNSTABLE_K = 3.0


class BreakableMonitor(PyATReadOnlyScalarVariable):
    """A monitor read that can be made to fail after the model is built.

    For the rollback path that only opens once the solve has *succeeded*: the
    model reads its outputs after solving, so a read failing there is the one
    way a write gets rejected with the simulator's cached solve already
    replaced.
    """

    broken: ClassVar[bool] = False

    def _get(self, simulator):
        if type(self).broken:
            raise RuntimeError("monitor read failed")
        return super()._get(simulator)


def build_variables():
    return [
        PyATWritableScalarVariable(
            name="quad", element_name="QUAD_F_01", attribute="K", default_value=QUAD_K
        ),
        PyATWritableScalarVariable(
            name="corrector",
            element_name="COR_H_03",
            attribute="KickAngle",
            index=0,
            default_value=0.0,
        ),
        PyATWritableScalarVariable(
            name="sextupole",
            element_name="SEXT_F_01",
            attribute="PolynomB",
            index=2,
            default_value=1.0,
        ),
        PyATReadOnlyScalarVariable(name="bpm_x", element_name="BPM_01", axis="x"),
        PyATReadOnlyScalarVariable(name="bpm_y", element_name="BPM_01", axis="y"),
    ]


@pytest.fixture
def simulator(test_ring):
    return PyATSimulator(test_ring)


@pytest.fixture
def model(simulator):
    return LUMEPyATModel(simulator=simulator, action_variables=build_variables())


def counting_solve(simulator, monkeypatch):
    """Wrap simulator.solve so a test can count how often a batch solves."""
    calls = []
    real = simulator.solve

    def counted():
        calls.append(None)
        return real()

    monkeypatch.setattr(simulator, "solve", counted)
    return calls


# -- construction ----------------------------------------------------------


def test_construction_registers_every_variable(model):
    assert len(model.supported_variables) == len(build_variables())
    assert sorted(model.supported_variables) == [
        "bpm_x",
        "bpm_y",
        "corrector",
        "quad",
        "sextupole",
    ]


def test_supported_variables_is_the_same_object_every_access(model):
    # The base property returns a fresh dict per call; callers may hold onto
    # this one, and get()/set() consult it on every call.
    assert model.supported_variables is model.supported_variables


def test_construction_validates_every_element_name(simulator):
    variables = build_variables()
    variables.append(
        PyATWritableScalarVariable(
            name="ghost", element_name="NOPE", attribute="K", default_value=1.0
        )
    )
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        LUMEPyATModel(simulator=simulator, action_variables=variables)


def test_construction_rejects_an_element_name_more_than_one_element_carries(simulator):
    # Every drift in the ring is named DRIFT. Binding that name addresses none
    # of them in particular, and element lookup would quietly pick the last.
    variables = build_variables()
    variables.append(
        PyATWritableScalarVariable(
            name="drift_length",
            element_name="DRIFT",
            attribute="Length",
            default_value=0.4,
        )
    )
    with pytest.raises(AmbiguousElementError, match="lattice elements are named"):
        LUMEPyATModel(simulator=simulator, action_variables=variables)


def test_duplicate_names_are_no_obstacle_when_no_variable_binds_them(simulator):
    # The same drifts, unaddressed: a model over this lattice is fine, and has
    # to stay fine -- a ring whose drifts share one name is the normal case.
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())

    assert sum(element.FamName == "DRIFT" for element in model.lattice) > 1
    model.set({"quad": 1.05})
    assert model.get("quad") == 1.05


def test_construction_rejects_an_attribute_the_element_does_not_have(simulator):
    # A typo'd attribute is a mistake in the *definition* of a variable, so it
    # is caught when the variable set is adopted rather than at whichever
    # later write happens to touch it. pyAT would otherwise accept the
    # assignment onto a dead field: ignored by the solve, read back intact.
    variables = build_variables()
    variables.append(
        PyATWritableScalarVariable(
            name="typo", element_name="QUAD_F_01", attribute="Kk", default_value=1.0
        )
    )
    with pytest.raises(AttributeError, match="'typo' declares attribute 'Kk'"):
        LUMEPyATModel(simulator=simulator, action_variables=variables)


def test_the_attribute_check_leaves_the_lattice_untouched(simulator):
    variables = [
        PyATWritableScalarVariable(
            name="typo", element_name="QUAD_F_01", attribute="Kk", default_value=1.0
        )
    ]
    with pytest.raises(AttributeError):
        LUMEPyATModel(simulator=simulator, action_variables=variables)

    assert not hasattr(simulator.element("QUAD_F_01"), "Kk")


def test_construction_rejects_a_variable_that_binds_no_element(simulator):
    # A lume-base writable that is not one of this package's kinds: it names
    # no element and no lattice, so the model cannot check or roll it back.
    class Unbound(WritableActionMixin[PyATSimulator], ScalarVariable):
        def _get(self, sim):
            return 0.0

        def _set(self, sim, value):
            pass

    with pytest.raises(TypeError, match="binds no lattice element"):
        LUMEPyATModel(
            simulator=simulator,
            action_variables=[Unbound(name="odd", default_value=1.0)],
        )


# -- multi-element bindings ------------------------------------------------


def pair(**overrides):
    kwargs = {
        "name": "pair",
        "bindings": [
            ElementBinding(element_name="QUAD_F_01", attribute="K"),
            ElementBinding(element_name="QUAD_F_02", attribute="K"),
        ],
        "default_value": QUAD_K,
    }
    return PyATWritableScalarVariable(**(kwargs | overrides))


def test_construction_validates_every_bound_element(simulator):
    variables = build_variables()
    variables.append(
        pair(
            bindings=[
                ElementBinding(element_name="QUAD_F_01", attribute="K"),
                ElementBinding(element_name="NOPE", attribute="K"),
            ]
        )
    )
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        LUMEPyATModel(simulator=simulator, action_variables=variables)


def test_construction_validates_every_bound_attribute(simulator):
    variables = build_variables()
    variables.append(
        pair(
            bindings=[
                ElementBinding(element_name="QUAD_F_01", attribute="K"),
                ElementBinding(element_name="QUAD_F_02", attribute="Kk"),
            ]
        )
    )
    with pytest.raises(AttributeError, match="'pair' declares attribute 'Kk'"):
        LUMEPyATModel(simulator=simulator, action_variables=variables)


def test_construction_rejects_an_index_beyond_the_attribute(simulator):
    # A quadrupole's PolynomB has two entries. Left to the write path, index
    # 5 would raise IndexError inside a batch -- rolled back, but only found
    # at the first write that touches it.
    variables = build_variables()
    variables.append(
        PyATWritableScalarVariable(
            name="octupole_term",
            element_name="QUAD_F_01",
            attribute="PolynomB",
            index=5,
            default_value=0.0,
        )
    )
    with pytest.raises(IndexError, match="'octupole_term' declares index 5"):
        LUMEPyATModel(simulator=simulator, action_variables=variables)


def test_a_multi_element_write_commits_every_element(simulator):
    model = LUMEPyATModel(simulator=simulator, action_variables=[pair()])
    model.set({"pair": 1.05})

    assert simulator.element("QUAD_F_01").K == pytest.approx(1.05)
    assert simulator.element("QUAD_F_02").K == pytest.approx(1.05)
    assert model.get("pair") == 1.05


# -- the lattice-level kind ------------------------------------------------


ENERGY = 1.0e9  # the test ring's energy, in eV


class EnergyWithRescale(PyATLatticeScalarVariable):
    """An energy variable that keeps one quadrupole's current fixed.

    The shape a facility's energy kind takes: a hook that rescales bound
    strengths after the energy is written -- K falls as 1/E at fixed current
    -- and a snapshot declaration that covers what the hook touches, so the
    rescale rolls back with the energy.
    """

    def _after_write(self, ring, value):
        index = ring.get_uint32_index("QUAD_F_01")[0]
        ring[index].K = QUAD_K * ENERGY / value

    def snapshot_targets(self, simulator):
        return [
            *super().snapshot_targets(simulator),
            (simulator.element_index("QUAD_F_01"), "K"),
        ]


def energy_variable(kind=PyATLatticeScalarVariable):
    return kind(name="energy", default_value=ENERGY)


def cavity_energies(ring):
    return [element.Energy for element in ring if isinstance(element, at.RFCavity)]


@pytest.fixture
def simulator_6d(test_ring_6d):
    return PyATSimulator(test_ring_6d)


def test_a_lattice_variable_is_registered_like_any_other(simulator_6d):
    variables = [
        *build_variables(),
        PyATLatticeScalarVariable(name="energy", default_value=ENERGY),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)

    assert "energy" in model.supported_variables
    assert model.get("energy") == ENERGY


def test_a_lattice_write_commits_the_ring_energy_and_every_cavity(simulator_6d):
    variables = [
        *build_variables(),
        PyATLatticeScalarVariable(name="energy", default_value=ENERGY),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)

    model.set({"energy": 1.1e9})

    assert simulator_6d.lattice.energy == pytest.approx(1.1e9)
    assert cavity_energies(simulator_6d.lattice) == [pytest.approx(1.1e9)]
    assert model.get("energy") == 1.1e9


def test_a_failed_batch_restores_the_ring_energy_and_every_cavity(
    simulator_6d, monkeypatch
):
    variables = [
        *build_variables(),
        PyATLatticeScalarVariable(name="energy", default_value=ENERGY),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)
    solution_before = simulator_6d.last_solution

    def fail():
        raise OrbitSolveError("forced")

    monkeypatch.setattr(simulator_6d, "solve", fail)

    with pytest.raises(OrbitSolveError, match="forced"):
        model.set({"energy": 1.1e9, "corrector": 1e-4})

    assert simulator_6d.lattice.energy == pytest.approx(ENERGY)
    assert cavity_energies(simulator_6d.lattice) == [pytest.approx(ENERGY)]
    assert list(simulator_6d.element("COR_H_03").KickAngle) == [0.0, 0.0]
    assert model.get("energy") == ENERGY
    assert simulator_6d.last_solution == solution_before


def test_the_hook_rescales_inside_the_same_batch(simulator_6d, monkeypatch):
    variables = [
        *build_variables(),
        EnergyWithRescale(name="energy", default_value=ENERGY),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)

    calls = counting_solve(simulator_6d, monkeypatch)
    model.set({"energy": 1.25e9})

    # One solve for the energy and the rescale together, and the rescaled
    # strength is what the lattice now carries.
    assert len(calls) == 1
    assert simulator_6d.element("QUAD_F_01").K == pytest.approx(QUAD_K / 1.25)
    # The quad's own variable still reports what it was last told: the
    # rescale is the energy kind's business, not a write of "quad".
    assert model.get("quad") == QUAD_K


def test_a_failed_batch_rolls_the_hooks_rescale_back(simulator_6d):
    variables = [
        *build_variables(),
        EnergyWithRescale(name="energy", default_value=ENERGY),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)

    # A third of the energy triples the quad's K, which destabilises the ring
    # -- the rescale is what fails the batch, and it must roll back with the
    # energy that caused it.
    with pytest.raises(OrbitSolveError, match="one-turn map unstable"):
        model.set({"energy": ENERGY / 3.0})

    assert simulator_6d.lattice.energy == pytest.approx(ENERGY)
    assert cavity_energies(simulator_6d.lattice) == [pytest.approx(ENERGY)]
    assert simulator_6d.element("QUAD_F_01").K == pytest.approx(QUAD_K)
    assert model.get("energy") == ENERGY


def test_rollback_restores_a_cavity_that_disagreed_with_the_ring(
    simulator_6d, monkeypatch
):
    # pyAT's Lattice.energy setter pushes the value onto every cavity. If the
    # rollback restored the ring after a cavity, that push would overwrite the
    # cavity's own snapshot -- visible whenever the two disagreed to begin
    # with, and the batch order put the cavity first. Lattice-level targets
    # are restored first for exactly this reason.
    cavity = simulator_6d.element("RF")
    cavity.Energy = 0.9e9  # set on the element only: the ring still says 1e9
    assert simulator_6d.lattice.energy == pytest.approx(ENERGY)

    variables = [
        PyATWritableScalarVariable(
            name="cavity_energy",
            element_name="RF",
            attribute="Energy",
            default_value=0.9e9,
        ),
        energy_variable(),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)

    def fail():
        raise OrbitSolveError("forced")

    monkeypatch.setattr(simulator_6d, "solve", fail)

    with pytest.raises(OrbitSolveError, match="forced"):
        model.set({"cavity_energy": 1.2e9, "energy": 1.1e9})

    assert simulator_6d.lattice.energy == pytest.approx(ENERGY)
    assert cavity.Energy == pytest.approx(0.9e9)


def test_construction_checks_a_lattice_variables_snapshot_targets(simulator_6d):
    # A subclass that declares a target the lattice does not have is a
    # mistake in its definition, caught where the variable set is adopted.
    class Typo(PyATLatticeScalarVariable):
        def snapshot_targets(self, simulator):
            return [*super().snapshot_targets(simulator), (None, "enrgy")]

    with pytest.raises(AttributeError, match="'energy' declares attribute 'enrgy'"):
        LUMEPyATModel(
            simulator=simulator_6d,
            action_variables=[Typo(name="energy", default_value=ENERGY)],
        )


def test_reset_writes_the_lattice_default_back(simulator_6d):
    variables = [
        *build_variables(),
        PyATLatticeScalarVariable(name="energy", default_value=ENERGY),
    ]
    model = LUMEPyATModel(simulator=simulator_6d, action_variables=variables)
    model.set({"energy": 1.1e9})

    model.reset()

    assert simulator_6d.lattice.energy == pytest.approx(ENERGY)
    assert cavity_energies(simulator_6d.lattice) == [pytest.approx(ENERGY)]
    assert model.get("energy") == ENERGY


def test_a_failed_batch_restores_every_bound_element(simulator):
    variables = [*build_variables(), pair()]
    model = LUMEPyATModel(simulator=simulator, action_variables=variables)
    before = model.get(["pair", "quad", "bpm_x"])

    with pytest.raises(OrbitSolveError):
        model.set({"pair": UNSTABLE_K, "corrector": 1e-4})

    assert simulator.element("QUAD_F_01").K == pytest.approx(QUAD_K)
    assert simulator.element("QUAD_F_02").K == pytest.approx(QUAD_K)
    assert list(simulator.element("COR_H_03").KickAngle) == [0.0, 0.0]
    assert model.get(["pair", "quad", "bpm_x"]) == before


def test_a_read_only_variable_bound_to_a_non_monitor_fails_at_construction(simulator):
    # The counterpart to solve.py's monitorless contract: solve_orbit does not
    # legislate how many monitors a ring should have, so this is where a
    # caller who expects readings finds out that it will not get them. The
    # element exists, so the element-name check passes -- the boot solve is
    # what catches it, and the message names the element.
    with pytest.raises(UnknownElementError, match="'QUAD_F_01' is not a monitor"):
        LUMEPyATModel(
            simulator=simulator,
            action_variables=[
                PyATReadOnlyScalarVariable(
                    name="not_a_bpm", element_name="QUAD_F_01", axis="x"
                )
            ],
        )


def test_a_writable_only_model_needs_no_monitors(test_ring):
    # Why the monitorless case is not an error: driving magnets and reading
    # setpoints back is a legitimate use with no monitors involved at all.
    model = LUMEPyATModel(
        simulator=PyATSimulator(strip_monitors(test_ring)),
        action_variables=[
            PyATWritableScalarVariable(
                name="quad",
                element_name="QUAD_F_01",
                attribute="K",
                default_value=QUAD_K,
            )
        ],
    )

    model.set({"quad": 1.05})
    assert model.get("quad") == 1.05


def test_construction_does_not_write_defaults_to_the_lattice(simulator):
    # The lattice is taken as built. A default that disagrees with it is
    # recorded, not applied -- reset() is how a caller reconciles the two.
    simulator.element("QUAD_F_01").K = 1.1

    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())

    assert simulator.element("QUAD_F_01").K == pytest.approx(1.1)
    assert model.get("quad") == pytest.approx(QUAD_K)

    model.reset()
    assert simulator.element("QUAD_F_01").K == pytest.approx(QUAD_K)


def test_construction_solves_the_boot_orbit_once(simulator, monkeypatch):
    calls = counting_solve(simulator, monkeypatch)
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())

    assert len(calls) == 1
    assert model.get(["bpm_x", "bpm_y"]) == {"bpm_x": 0.0, "bpm_y": 0.0}


def test_construction_on_an_unstable_lattice_raises(simulator):
    for index in simulator.lattice.get_uint32_index("QUAD_F_*"):
        simulator.lattice[index].K = UNSTABLE_K

    with pytest.raises(OrbitSolveError, match="one-turn map unstable"):
        LUMEPyATModel(simulator=simulator, action_variables=build_variables())


# -- the frozen variable set -----------------------------------------------


def test_registering_after_construction_raises(model):
    extra = PyATWritableScalarVariable(
        name="late", element_name="QUAD_D_01", attribute="K", default_value=-QUAD_K
    )
    with pytest.raises(NotImplementedError, match="fixed at construction"):
        model.register_action_variable(extra)


def test_unregistering_after_construction_raises(model):
    with pytest.raises(NotImplementedError, match="fixed at construction"):
        model.unregister_action_variable("quad")


def test_a_failed_mutation_leaves_the_variable_set_intact(model):
    with pytest.raises(NotImplementedError):
        model.unregister_action_variable("quad")
    assert "quad" in model.supported_variables


# -- reads -----------------------------------------------------------------


def test_writables_read_back_bit_exactly(model):
    value = 1.2345678901234567
    model.set({"quad": value})
    # Served from the retained value, not re-derived from the lattice.
    assert model.get("quad") == value


def test_read_only_variables_read_the_committed_solve(model, simulator):
    model.set({"corrector": 1e-4})
    solution = simulator.last_solution

    assert model.get("bpm_x") == pytest.approx(solution["BPM_01"][0])
    assert model.get("bpm_y") == pytest.approx(solution["BPM_01"][1])


def test_reading_an_unknown_name_raises(model):
    with pytest.raises(UnknownElementError, match="'nope' is not a variable"):
        model._get(["nope"])


# -- writes ----------------------------------------------------------------


def test_a_batch_solves_exactly_once(model, simulator, monkeypatch):
    calls = counting_solve(simulator, monkeypatch)
    model.set({"quad": 1.1, "corrector": 1e-5, "sextupole": 1.5})

    assert len(calls) == 1


def test_a_batch_applies_every_value(model, simulator):
    model.set({"quad": 1.1, "corrector": 1e-5, "sextupole": 1.5})

    assert simulator.element("QUAD_F_01").K == pytest.approx(1.1)
    assert simulator.element("COR_H_03").KickAngle[0] == pytest.approx(1e-5)
    assert simulator.element("SEXT_F_01").PolynomB[2] == pytest.approx(1.5)


def test_writing_an_unknown_name_raises_before_anything_is_written(model, simulator):
    with pytest.raises(UnknownElementError, match="'nope' is not a variable"):
        model._set({"quad": 1.1, "nope": 1.0})

    assert simulator.element("QUAD_F_01").K == pytest.approx(QUAD_K)


def test_writing_a_read_only_variable_raises_before_anything_is_written(
    model, simulator
):
    # The base _set silently skips non-writables, so without validating first
    # this would look like a successful no-op.
    with pytest.raises(UnknownElementError, match="'bpm_x' is not a settable input"):
        model._set({"quad": 1.1, "bpm_x": 1.0})

    assert simulator.element("QUAD_F_01").K == pytest.approx(QUAD_K)


def test_the_public_api_reports_a_read_only_write_as_such(model):
    with pytest.raises(ReadOnlyError, match="read-only"):
        model.set({"bpm_x": 1.0})


# -- atomicity -------------------------------------------------------------


def test_a_failed_batch_restores_every_element(model, simulator):
    before = model.get(["quad", "corrector", "sextupole", "bpm_x", "bpm_y"])
    lattice_before = (
        simulator.element("QUAD_F_01").K,
        list(simulator.element("COR_H_03").KickAngle),
        list(simulator.element("SEXT_F_01").PolynomB),
    )
    solution_before = simulator.last_solution

    # The quad value alone destabilises the ring; the other two are benign.
    with pytest.raises(OrbitSolveError):
        model.set({"quad": UNSTABLE_K, "corrector": 1e-4, "sextupole": 2.0})

    assert simulator.element("QUAD_F_01").K == lattice_before[0]
    assert list(simulator.element("COR_H_03").KickAngle) == lattice_before[1]
    assert list(simulator.element("SEXT_F_01").PolynomB) == lattice_before[2]
    assert model.get(["quad", "corrector", "sextupole", "bpm_x", "bpm_y"]) == before
    assert simulator.last_solution == solution_before


def test_rollback_covers_an_arbitrary_bound_attribute(simulator, monkeypatch):
    # Nothing about the rollback is specific to strength fields: it snapshots
    # whatever attribute a variable declares.
    variables = build_variables()
    variables.append(
        PyATWritableScalarVariable(
            name="length",
            element_name="QUAD_F_02",
            attribute="Length",
            default_value=0.3,
        )
    )
    model = LUMEPyATModel(simulator=simulator, action_variables=variables)

    before = model.get(["length", "bpm_x"])
    length_before = simulator.element("QUAD_F_02").Length
    solution_before = simulator.last_solution

    def fail():
        raise OrbitSolveError("forced")

    monkeypatch.setattr(simulator, "solve", fail)

    with pytest.raises(OrbitSolveError, match="forced"):
        model.set({"length": 0.55})

    assert simulator.element("QUAD_F_02").Length == length_before
    assert model.get(["length", "bpm_x"]) == before
    assert simulator.last_solution == solution_before


def test_rollback_restores_a_whole_sequence_not_just_the_written_slot(
    simulator, monkeypatch
):
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())
    before = list(simulator.element("SEXT_F_01").PolynomB)

    def fail():
        raise OrbitSolveError("forced")

    monkeypatch.setattr(simulator, "solve", fail)

    with pytest.raises(OrbitSolveError):
        model.set({"sextupole": 9.9})

    assert list(simulator.element("SEXT_F_01").PolynomB) == before


def test_a_failing_variable_rolls_back_the_rest_of_its_batch(simulator, monkeypatch):
    # A failure part-way through the dispatch, not at the solve.
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())
    before = simulator.element("QUAD_F_01").K

    original = PyATWritableScalarVariable._set

    def explode(self, sim, value):
        if self.name == "sextupole":
            raise RuntimeError("boom")
        original(self, sim, value)

    monkeypatch.setattr(PyATWritableScalarVariable, "_set", explode)

    with pytest.raises(RuntimeError, match="boom"):
        model.set({"quad": 1.1, "sextupole": 2.0})

    assert simulator.element("QUAD_F_01").K == before


def test_a_failure_after_the_solve_still_rolls_the_solve_back(simulator, monkeypatch):
    # The one path where the lattice alone is not enough: the solve succeeds
    # and replaces the simulator's cached reading, and only then does reading
    # the outputs fail. Restoring the lattice without restoring the cache
    # would leave last_solution describing an orbit the ring no longer has.
    variables = build_variables()
    variables.append(BreakableMonitor(name="flaky", element_name="BPM_02", axis="x"))
    model = LUMEPyATModel(simulator=simulator, action_variables=variables)

    kick_before = list(simulator.element("COR_H_03").KickAngle)
    inputs_before = model.get(["quad", "corrector"])
    solution_before = dict(simulator.last_solution)

    monkeypatch.setattr(BreakableMonitor, "broken", True)
    with pytest.raises(RuntimeError, match="monitor read failed"):
        model.set({"corrector": 1e-4})

    assert list(simulator.element("COR_H_03").KickAngle) == kick_before
    assert model.get(["quad", "corrector"]) == inputs_before
    assert simulator.last_solution == solution_before


def test_a_successful_batch_commits(model, simulator):
    model.set({"quad": 1.1, "corrector": 1e-5})

    assert model.get(["quad", "corrector"]) == {"quad": 1.1, "corrector": 1e-5}
    assert simulator.element("QUAD_F_01").K == pytest.approx(1.1)


# -- reset -----------------------------------------------------------------


def test_reset_restores_every_default_in_one_solve(model, simulator, monkeypatch):
    model.set({"quad": 1.1, "corrector": 1e-5, "sextupole": 1.5})

    calls = counting_solve(simulator, monkeypatch)
    model.reset()

    assert len(calls) == 1
    assert model.get(["quad", "corrector", "sextupole"]) == {
        "quad": QUAD_K,
        "corrector": 0.0,
        "sextupole": 1.0,
    }


def test_reset_preserves_construction_time_misalignments(test_ring):
    simulator = PyATSimulator(
        test_ring, element_misalignments={"QUAD_D_01": {"dy": 300e-6}}
    )
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())
    misaligned = model.get(["bpm_x", "bpm_y"])

    model.set({"corrector": 1e-4})
    model.reset()

    # reset undoes writes, not faults: the lattice is never rebuilt.
    assert model.get(["bpm_x", "bpm_y"]) == misaligned
    assert test_ring[test_ring.get_uint32_index("QUAD_D_01")[0]].T1[2] != 0.0


def test_a_failing_reset_leaves_the_lattice_as_it_was(simulator, monkeypatch):
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())
    model.set({"quad": 1.1})
    before = simulator.element("QUAD_F_01").K

    def fail():
        raise OrbitSolveError("forced")

    monkeypatch.setattr(simulator, "solve", fail)

    with pytest.raises(OrbitSolveError):
        model.reset()

    assert simulator.element("QUAD_F_01").K == before
    assert model.get("quad") == pytest.approx(1.1)


# -- accessors -------------------------------------------------------------


def test_the_model_exposes_its_lattice_and_element_lookup(model, test_ring):
    assert model.lattice is test_ring
    assert test_ring[model.element_index("QUAD_F_01")].FamName == "QUAD_F_01"


def test_element_index_rejects_an_unknown_name(model):
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        model.element_index("NOPE")


def test_an_unstable_write_never_exits_the_process(model):
    # Whether an unusable model should end the process is the caller's call.
    try:
        model.set({"quad": UNSTABLE_K})
    except SystemExit:  # pragma: no cover - the point of the test
        pytest.fail("set() raised SystemExit")
    except OrbitSolveError:
        pass


def test_the_model_works_against_a_lattice_it_did_not_build():
    # Nothing here builds or owns a lattice; the caller supplies one.
    simulator = PyATSimulator(build_test_ring())
    model = LUMEPyATModel(simulator=simulator, action_variables=build_variables())
    assert model.get("bpm_x") == 0.0
