"""Typed action variables binding lume-base names to pyAT element attributes.

Values are always in native pyAT units; unit conversion and any other
transformer logic belongs to the layer that builds these variables.

Variable names are free-form. Nothing here parses them, derives meaning from
them, or requires them to follow any convention — a name is whatever the
caller's own address space calls this quantity. What binds a variable to the
lattice is its bindings — element name plus attribute — never the name.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import at
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.exceptions import ReadOnlyError
from lume.variables import ScalarVariable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lume_pyat.exceptions import UnknownElementError
from lume_pyat.simulator import PyATSimulator

__all__ = [
    "ElementBinding",
    "PyATLatticeScalarVariable",
    "PyATReadOnlyScalarVariable",
    "PyATWritableScalarVariable",
]

# Where a snapshot lives: an element index and the attribute on it, or
# ``None`` and an attribute of the lattice itself.
SnapshotTarget = tuple[int | None, str]


def _require_a_default_value(variable: ScalarVariable) -> None:
    """Reject a writable with no default, at definition time.

    ``ActionModel.reset()`` writes every writable's ``default_value`` back to
    the simulator. A variable without one would push ``None`` into the lattice
    on reset, so the failure is moved forward to where the variable is
    declared and the mistake is visible.
    """
    if variable.default_value is None:
        raise ValueError(
            f"writable variable {variable.name!r} needs a float default_value: "
            "it is what reset() writes back to the lattice"
        )


class ElementBinding(BaseModel):
    """One element attribute a writable variable drives, and by how much.

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
    weight : float
        What the element receives per unit of the variable's value: a write of
        ``v`` puts ``v * weight`` on this element. Finite and nonzero, because
        the read divides by it. ``1.0`` binds the value as is.
    """

    # A plain record: a misspelled field is a mistake, not an extra to ignore.
    model_config = ConfigDict(extra="forbid")

    element_name: str
    attribute: str
    index: int | None = None
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _require_a_usable_weight(cls, weight: float) -> float:
        if not math.isfinite(weight) or weight == 0.0:
            raise ValueError(
                f"weight must be finite and nonzero, got {weight!r}: "
                "the read divides the element's value by it"
            )
        return weight


class PyATWritableScalarVariable(WritableActionMixin[PyATSimulator], ScalarVariable):
    """A settable scalar driving one attribute on one or more lattice elements.

    This is the extension point for a facility layer. Bind a variable straight
    to an element attribute for a raw, native-unit control, or subclass it and
    override :meth:`_set` / :meth:`_get` to put a conversion in between —
    engineering units to strength, a calibration curve, a shared lookup table.
    Whatever that conversion is, it belongs in the subclass, not here: this
    package knows only about pyAT quantities.

    A variable carries a list of :class:`ElementBinding`. A write of ``v``
    puts ``v * weight`` on every bound element; a read returns the first
    element's value divided by its weight. One setpoint driving several
    lattice slices — each slice a fraction of a kick, or every slice the same
    strength — is one variable with one binding per slice.

    The single-element form is a shorthand: ``element_name=``, ``attribute=``
    and optionally ``index=`` build one binding with unit weight. It is the
    same variable as ``bindings=[ElementBinding(...)]``.

    Attributes
    ----------
    bindings : list[ElementBinding]
        The elements this variable drives, in order. At least one. The first
        is the one a read comes from.
    """

    bindings: list[ElementBinding] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _expand_the_single_element_form(cls, data: Any) -> Any:
        """Turn ``element_name``/``attribute``/``index`` into one binding."""
        if not isinstance(data, dict) or "element_name" not in data:
            return data
        if "bindings" in data:
            raise ValueError(
                "give either bindings= or the single-element form "
                "(element_name=, attribute=, index=), not both"
            )
        data = dict(data)
        binding = {
            "element_name": data.pop("element_name"),
            "attribute": data.pop("attribute", None),
            "index": data.pop("index", None),
        }
        data["bindings"] = [ElementBinding.model_validate(binding)]
        return data

    @model_validator(mode="after")
    def _check_default_value(self) -> PyATWritableScalarVariable:
        _require_a_default_value(self)
        return self

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:
        """Every ``(element index, attribute)`` a write of this variable touches.

        What :class:`~lume_pyat.model.LUMEPyATModel` snapshots before a batch
        and restores if the batch fails: one entry per binding.

        Raises
        ------
        UnknownElementError
            A binding names an element the lattice does not have.
        """
        return [
            (simulator.element_index(binding.element_name), binding.attribute)
            for binding in self.bindings
        ]

    def _get(self, simulator: PyATSimulator) -> float:
        """Read the first binding's element, in native pyAT units.

        The element holds ``value * weight``, so the value is the element's
        reading divided by the weight.
        """
        first = self.bindings[0]
        value = getattr(simulator.element(first.element_name), first.attribute)
        if first.index is not None:
            value = value[first.index]
        return float(value) / first.weight

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Write ``value * weight`` to every bound element, in native pyAT units.

        Every element is checked before any is written, so a bad binding
        leaves the lattice untouched.

        Raises
        ------
        AttributeError
            An element has no such attribute. A variable used through
            :class:`~lume_pyat.model.LUMEPyATModel` cannot reach this: the
            model checks the same condition when it adopts the variable, so
            the mistake surfaces at construction. This is the backstop for a
            variable driven against a simulator directly, where there is no
            construction step to catch it.
        """
        elements = [
            simulator.element(binding.element_name) for binding in self.bindings
        ]
        for binding, element in zip(self.bindings, elements, strict=True):
            if not hasattr(element, binding.attribute):
                # pyAT elements accept arbitrary attribute assignment, so a
                # typo here would otherwise be written to a dead attribute,
                # ignored by the solve, and read back intact -- a silent
                # no-op write.
                raise AttributeError(
                    f"element {binding.element_name!r} has no attribute "
                    f"{binding.attribute!r} to write"
                )
        for binding, element in zip(self.bindings, elements, strict=True):
            weighted = value * binding.weight
            if binding.index is None:
                setattr(element, binding.attribute, weighted)
            else:
                getattr(element, binding.attribute)[binding.index] = weighted


class PyATLatticeScalarVariable(WritableActionMixin[PyATSimulator], ScalarVariable):
    """The ring energy as a settable scalar, in eV — the lattice-level kind.

    Where :class:`PyATWritableScalarVariable` binds element attributes, this
    kind binds the lattice as a whole: a write sets ``ring.energy`` and the
    ``Energy`` of every ``at.RFCavity``, and a read returns ``ring.energy``.
    Through :class:`~lume_pyat.model.LUMEPyATModel` it gets the same
    all-or-nothing treatment as any other write: the ring energy and every
    cavity energy are snapshotted before the batch and restored if it fails.

    Like the element kind it is an extension point. A facility's energy
    variable subclasses it and overrides :meth:`_set` / :meth:`_get` to
    convert its own units — a dipole current, say — to and from eV. Changing
    the energy does not change the normalised strengths pyAT stores, so a
    facility that holds magnet *currents* fixed across an energy change
    rescales those strengths itself, in :meth:`_after_write`, which runs
    inside the same batch as the energy write: one solve, one rollback. A
    subclass that touches elements there also extends
    :meth:`snapshot_targets` with what it touches, so the rollback covers
    them.

    Radiation-on rings are out of scope: nothing here accounts for the
    energy loss per turn.
    """

    @model_validator(mode="after")
    def _check_default_value(self) -> PyATLatticeScalarVariable:
        _require_a_default_value(self)
        return self

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:
        """Everything a write of this variable touches: the ring energy and
        the ``Energy`` of every ``at.RFCavity``, in ring order.

        A subclass whose :meth:`_after_write` touches other attributes
        returns those too, appended to ``super().snapshot_targets(...)``.
        """
        ring = simulator.lattice
        return [
            (None, "energy"),
            *(
                (index, "Energy")
                for index, element in enumerate(ring)
                if isinstance(element, at.RFCavity)
            ),
        ]

    def _get(self, simulator: PyATSimulator) -> float:
        """Read the ring energy, in eV."""
        return float(simulator.lattice.energy)

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Write ``value`` as the ring energy and every cavity's ``Energy``,
        then run :meth:`_after_write`."""
        ring = simulator.lattice
        ring.energy = value
        for element in ring:
            if isinstance(element, at.RFCavity):
                element.Energy = value
        self._after_write(ring, value)

    def _after_write(self, ring: at.Lattice, value: float) -> None:
        """Hook: runs after the energy is written, inside the same batch.

        A no-op here. A subclass that keeps other lattice quantities
        consistent with the energy — bound strengths at fixed current, for
        instance — does so here, on ``ring``, and declares what it touches in
        :meth:`snapshot_targets`.

        Args:
            ring: The live lattice, already carrying the new energy.
            value: The energy just written, in eV.
        """


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
