"""LUMEPyATModel: construction, the frozen variable set, and atomic writes."""

import at
import pytest
from lume.exceptions import ReadOnlyError

from lume_pyat.actions import PyATReadOnlyScalarVariable, PyATWritableScalarVariable
from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator
from lume_pyat.tests.conftest import QUAD_K, build_test_ring

UNSTABLE_K = 3.0


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


def test_construction_rejects_a_variable_that_binds_no_element(simulator):
    class Unbound(PyATWritableScalarVariable):
        element_name: str | None = None

    with pytest.raises(TypeError, match="binds no lattice element"):
        LUMEPyATModel(
            simulator=simulator,
            action_variables=[
                Unbound(name="odd", attribute="K", default_value=1.0),
            ],
        )


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
    monitorless = at.Lattice(
        [element for element in test_ring if not isinstance(element, at.Monitor)],
        name="NO_MONITORS",
        energy=test_ring.energy,
        periodicity=1,
    )
    monitorless.disable_6d()
    model = LUMEPyATModel(
        simulator=PyATSimulator(monitorless),
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

    with pytest.raises(OrbitSolveError, match="one-turn matrix unstable"):
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
