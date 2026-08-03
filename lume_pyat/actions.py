"""Typed action variables binding lume-base names to pyAT element attributes.

Values are always in native pyAT units; unit conversion and any other
transformer logic belongs to the layer that builds these variables.

Variable names are free-form. Nothing here parses them, derives meaning from
them, or requires them to follow any convention — a name is whatever the
caller's own address space calls this quantity. What binds a variable to the
lattice is ``element_name`` plus the attribute it declares, never the name.
"""

from __future__ import annotations

from typing import Literal

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.exceptions import ReadOnlyError
from lume.variables import ScalarVariable
from pydantic import model_validator

from lume_pyat.exceptions import UnknownElementError
from lume_pyat.simulator import PyATSimulator

__all__ = [
    "PyATReadOnlyScalarVariable",
    "PyATWritableScalarVariable",
]


class PyATWritableScalarVariable(WritableActionMixin[PyATSimulator], ScalarVariable):
    """A settable scalar bound to one attribute of one lattice element.

    This is the extension point for a facility layer. Bind a variable straight
    to an element attribute for a raw, native-unit control, or subclass it and
    override :meth:`_set` / :meth:`_get` to put a conversion in between —
    engineering units to strength, a calibration curve, a shared lookup table.
    Whatever that conversion is, it belongs in the subclass, not here: this
    package knows only about pyAT quantities.

    Attributes
    ----------
    element_name : str
        ``FamName`` of the target element.
    attribute : str
        The element attribute to read and write, e.g. ``"K"`` or
        ``"PolynomB"``.
    index : int | None
        Position within ``attribute`` when it holds a sequence, e.g. ``1`` for
        the quadrupole term of ``PolynomB``. ``None`` for a scalar attribute.
    """

    element_name: str
    attribute: str
    index: int | None = None

    @model_validator(mode="after")
    def _require_a_default_value(self) -> PyATWritableScalarVariable:
        """Reject a writable with no default, at definition time.

        ``ActionModel.reset()`` writes every writable's ``default_value`` back
        to the simulator. A variable without one would push ``None`` into the
        lattice on reset, so the failure is moved forward to where the variable
        is declared and the mistake is visible.
        """
        if self.default_value is None:
            raise ValueError(
                f"writable variable {self.name!r} needs a float default_value: "
                "it is what reset() writes back to the lattice"
            )
        return self

    def _get(self, simulator: PyATSimulator) -> float:
        """Read the bound attribute, in native pyAT units."""
        value = getattr(simulator.element(self.element_name), self.attribute)
        if self.index is not None:
            value = value[self.index]
        return float(value)

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Write the bound attribute, in native pyAT units."""
        element = simulator.element(self.element_name)
        if not hasattr(element, self.attribute):
            # pyAT elements accept arbitrary attribute assignment, so a typo
            # here would otherwise be written to a dead attribute, ignored by
            # the solve, and read back intact -- a silent no-op write.
            raise AttributeError(
                f"element {self.element_name!r} has no attribute "
                f"{self.attribute!r} to write"
            )
        if self.index is None:
            setattr(element, self.attribute, value)
        else:
            getattr(element, self.attribute)[self.index] = value


class PyATReadOnlyScalarVariable(ReadOnlyActionMixin[PyATSimulator], ScalarVariable):
    """One transverse coordinate of the solved orbit at one monitor.

    Reads come from :attr:`~lume_pyat.simulator.PyATSimulator.last_solution`,
    not from a fresh solve, so a consumer reading every monitor pays for one
    solve rather than one per variable — and every reading it collects belongs
    to the same orbit.

    Attributes
    ----------
    element_name : str
        ``FamName`` of the monitor to read.
    axis : {"x", "y"}
        Which transverse coordinate of that monitor's reading to return.
    """

    element_name: str
    axis: Literal["x", "y"]
    read_only: bool = True

    def _get(self, simulator: PyATSimulator) -> float:
        """Read this monitor's coordinate off the last successful solve.

        Raises
        ------
        OrbitSolveError
            No solve has succeeded yet.
        UnknownElementError
            ``element_name`` is not a monitor of this lattice.
        """
        solution = simulator.last_solution
        try:
            reading = solution[self.element_name]
        except KeyError:
            raise UnknownElementError(
                f"{self.element_name!r} is not a monitor of this lattice"
            ) from None
        return float(reading[0] if self.axis == "x" else reading[1])

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Always raises — a solved orbit is not settable.

        The public ``set()`` path rejects a read-only variable before reaching
        here; this makes a direct call fail the same way rather than with an
        ``AttributeError``.
        """
        raise ReadOnlyError(f"variable {self.name!r} is read-only and cannot be set")
