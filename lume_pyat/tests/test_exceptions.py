"""The exception contract: base classes, message wording, and import weight."""

import subprocess
import sys

import pytest

import lume_pyat
from lume_pyat.exceptions import OrbitSolveError, UnknownElementError

# Stricter than the package-level guarantee: this module must not reach numpy
# either, so consumers can re-export it from anywhere.
FORBIDDEN_ROOTS = ("at", "lume", "numpy", "h5py")


def test_import_is_standard_library_only():
    probe = (
        "import sys, lume_pyat.exceptions; "
        f"roots = {FORBIDDEN_ROOTS!r}; "
        "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] in roots)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_orbit_solve_error_is_a_plain_exception():
    # An unstable lattice is a failed solve, not a bad argument.
    assert issubclass(OrbitSolveError, Exception)
    assert not issubclass(OrbitSolveError, ValueError)


def test_unknown_element_error_is_a_value_error():
    # Consumers alias this class; an existing `except ValueError` must still catch it.
    assert issubclass(UnknownElementError, ValueError)
    with pytest.raises(ValueError):
        raise UnknownElementError("anything")


def test_for_name_produces_the_contract_message():
    error = UnknownElementError.for_name("QUAD_F_01")
    assert str(error) == "no lattice element named 'QUAD_F_01'"
    assert isinstance(error, UnknownElementError)


def test_for_name_survives_subclassing():
    # Consumers subclass or alias; the classmethod must build their type.
    class Derived(UnknownElementError):
        pass

    assert type(Derived.for_name("X")) is Derived


def test_constructor_leaves_the_message_alone():
    # Aliasing consumers raise this for their own lookup failures.
    assert str(UnknownElementError("'QF:99' is not a variable of this model")) == (
        "'QF:99' is not a variable of this model"
    )


def test_lazy_package_access_yields_the_same_classes():
    # Class identity is load-bearing: consumers alias these names and pin
    # identity, not the class name.
    assert lume_pyat.OrbitSolveError is OrbitSolveError
    assert lume_pyat.UnknownElementError is UnknownElementError
