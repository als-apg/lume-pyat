"""Action variables: attribute binding, the default-value rule, read-only paths."""

import pytest
from lume.actions import (
    Action,
    ActionModel,
    ReadOnlyActionMixin,
    WritableActionMixin,
)
from lume.exceptions import ReadOnlyError
from lume.variables import ScalarVariable
from pydantic import ValidationError

from lume_pyat.actions import PyATReadOnlyScalarVariable, PyATWritableScalarVariable
from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.simulator import PyATSimulator
from lume_pyat.tests.conftest import QUAD_K


@pytest.fixture
def simulator(test_ring):
    return PyATSimulator(test_ring)


def writable(**overrides):
    kwargs = {
        "name": "quad_strength",
        "element_name": "QUAD_F_01",
        "attribute": "K",
        "default_value": QUAD_K,
    }
    return PyATWritableScalarVariable(**(kwargs | overrides))


def readonly(**overrides):
    kwargs = {"name": "bpm_x", "element_name": "BPM_01", "axis": "x"}
    return PyATReadOnlyScalarVariable(**(kwargs | overrides))


# -- the lume-base contract ------------------------------------------------


def test_both_classes_are_actions_and_scalar_variables():
    assert isinstance(writable(), (Action, ScalarVariable))
    assert isinstance(readonly(), (Action, ScalarVariable))
    assert isinstance(writable(), WritableActionMixin)
    assert isinstance(readonly(), ReadOnlyActionMixin)


def test_a_writable_is_not_read_only_by_default():
    assert writable().read_only is False


def test_a_read_only_variable_defaults_to_read_only():
    assert readonly().read_only is True


def test_a_read_only_variable_cannot_be_declared_writable():
    with pytest.raises(ReadOnlyError, match="requires read_only=True"):
        readonly(read_only=False)


# -- names are free-form ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["SR:MAG:QF:01:CURRENT:SP", "quad 1 strength", "x", "汉字", "1234"],
)
def test_variable_names_follow_no_grammar(name):
    # Nothing here parses names or derives meaning from them; binding is by
    # element_name and attribute alone.
    assert writable(name=name).name == name


# -- the default-value rule ------------------------------------------------


def test_a_writable_requires_a_default_value():
    with pytest.raises(ValidationError, match="needs a float default_value"):
        PyATWritableScalarVariable(
            name="no_default", element_name="QUAD_F_01", attribute="K"
        )


def test_a_writable_default_value_is_coerced_to_float():
    assert isinstance(writable(default_value=2).default_value, float)


def test_a_read_only_variable_needs_no_default_value():
    # Nothing writes a read-only variable back, so reset() never reads this.
    assert readonly().default_value is None


# -- writable get/set ------------------------------------------------------


def test_a_writable_reads_the_bound_attribute(simulator):
    assert writable()._get(simulator) == pytest.approx(QUAD_K)


def test_a_writable_writes_the_bound_attribute(simulator):
    variable = writable()
    variable._set(simulator, 1.75)

    assert simulator.element("QUAD_F_01").K == pytest.approx(1.75)
    assert variable._get(simulator) == pytest.approx(1.75)


def test_an_indexed_writable_addresses_one_slot(simulator):
    sextupole = writable(
        name="sext", element_name="SEXT_F_01", attribute="PolynomB", index=2
    )
    before = list(simulator.element("SEXT_F_01").PolynomB)

    sextupole._set(simulator, 2.5)

    after = list(simulator.element("SEXT_F_01").PolynomB)
    assert after[2] == pytest.approx(2.5)
    assert after[:2] == before[:2]
    assert sextupole._get(simulator) == pytest.approx(2.5)


def test_a_writable_uses_native_units_without_conversion(simulator):
    # The value written is the value pyAT stores, byte for byte.
    variable = writable()
    variable._set(simulator, 1.2345678901234567)
    assert simulator.element("QUAD_F_01").K == 1.2345678901234567


def test_a_writable_rejects_an_unknown_element(simulator):
    variable = writable(element_name="NOPE")
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        variable._set(simulator, 1.0)
    with pytest.raises(UnknownElementError, match="no lattice element named 'NOPE'"):
        variable._get(simulator)


def test_a_writable_rejects_an_attribute_the_element_does_not_have(simulator):
    # pyAT elements accept arbitrary assignment, so an unguarded write to a
    # misspelled attribute would be a silent no-op the solve ignores.
    variable = writable(attribute="Kk")
    with pytest.raises(AttributeError, match="has no attribute 'Kk'"):
        variable._set(simulator, 1.0)
    assert not hasattr(simulator.element("QUAD_F_01"), "Kk")


def test_a_writable_is_the_facility_extension_point(simulator):
    # Subclassing to insert a conversion is the documented pattern.
    class AmpsVariable(PyATWritableScalarVariable):
        amps_per_unit: float

        def _set(self, sim, value):
            super()._set(sim, value / self.amps_per_unit)

        def _get(self, sim):
            return super()._get(sim) * self.amps_per_unit

    variable = AmpsVariable(
        name="quad_current",
        element_name="QUAD_F_01",
        attribute="K",
        default_value=QUAD_K,
        amps_per_unit=100.0,
    )
    variable._set(simulator, 150.0)

    assert simulator.element("QUAD_F_01").K == pytest.approx(1.5)
    assert variable._get(simulator) == pytest.approx(150.0)


# -- read-only get ---------------------------------------------------------


def test_a_read_only_variable_reads_the_last_solution(simulator):
    simulator.element("COR_H_03").KickAngle = [1e-4, 0.0]
    solution = simulator.solve()

    assert readonly()._get(simulator) == pytest.approx(solution["BPM_01"][0])
    assert readonly(axis="y")._get(simulator) == pytest.approx(solution["BPM_01"][1])


def test_a_read_only_variable_does_not_re_solve(simulator, monkeypatch):
    simulator.solve()

    def fail(*args, **kwargs):
        raise AssertionError("_get must not trigger a solve")

    monkeypatch.setattr(PyATSimulator, "solve", fail)
    assert readonly()._get(simulator) == pytest.approx(0.0)


def test_reads_before_any_solve_are_a_clear_error(simulator):
    with pytest.raises(OrbitSolveError, match="no closed orbit has been solved yet"):
        readonly()._get(simulator)


def test_a_read_only_variable_rejects_a_non_monitor_element(simulator):
    simulator.solve()
    with pytest.raises(UnknownElementError, match="is not a monitor of this lattice"):
        readonly(element_name="QUAD_F_01")._get(simulator)


def test_the_axis_must_be_x_or_y():
    with pytest.raises(ValidationError):
        readonly(axis="z")


def test_setting_a_read_only_variable_raises(simulator):
    with pytest.raises(ReadOnlyError, match="is read-only"):
        readonly()._set(simulator, 1.0)


def test_read_only_error_is_a_type_error():
    # Consumers catch it; pinning the base class keeps that catch valid.
    assert issubclass(ReadOnlyError, TypeError)


# -- against real lume-base dispatch ---------------------------------------
#
# These pin the contract against the installed lume-base rather than against
# its documentation. They use ActionModel directly, so they stay meaningful
# independently of anything this package builds on top of it.


@pytest.fixture
def model(simulator):
    return ActionModel(simulator, [writable(), readonly()])


def test_a_writable_round_trips_through_the_public_api(model, simulator):
    model.set({"quad_strength": 1.3})

    assert simulator.element("QUAD_F_01").K == pytest.approx(1.3)
    assert model.get(["quad_strength"]) == {"quad_strength": 1.3}


def test_setting_a_read_only_variable_through_the_public_api_raises(model):
    with pytest.raises(ReadOnlyError, match="read-only"):
        model.set({"bpm_x": 1.0})


def test_a_read_only_variable_reads_through_the_public_api(model, simulator):
    simulator.element("COR_H_03").KickAngle = [1e-4, 0.0]
    solution = simulator.solve()

    assert model.get(["bpm_x"]) == {"bpm_x": pytest.approx(solution["BPM_01"][0])}


def test_reset_writes_every_writable_default_back(model, simulator):
    model.set({"quad_strength": 1.3})
    model.reset()

    assert simulator.element("QUAD_F_01").K == pytest.approx(QUAD_K)
