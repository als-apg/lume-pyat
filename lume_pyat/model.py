"""``LUMEPyATModel`` — the lume-base ``ActionModel`` implementation for pyAT."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from lume.actions import ActionModel, ActionVariable, WritableActionMixin
from lume.variables import Variable

from lume_pyat.actions import ElementBinding, PyATLatticeScalarVariable
from lume_pyat.exceptions import UnknownElementError
from lume_pyat.simulator import PyATSimulator

if TYPE_CHECKING:  # pragma: no cover
    import at

__all__ = ["LUMEPyATModel"]

# One entry per attribute a batch will touch: where it lives (an element
# index, or None for the lattice itself), and what it held before the batch
# started.
_Snapshot = tuple[int | None, str, Any]


class LUMEPyATModel(ActionModel[PyATSimulator]):
    """A lume-base model over one pyAT lattice, with atomic multi-set.

    What this adds over the plain ``ActionModel`` contract is that a
    multi-variable write either lands completely or not at all. Every element
    the batch will touch is snapshotted first; the whole batch is applied; the
    orbit is solved **once**; and only then are the retained inputs and cached
    outputs committed. If anything fails — an unstable orbit, a conversion
    raising inside a subclassed variable, anything at all — every snapshot is
    restored, the simulator's cached solve is put back, and the model is left
    exactly as it was. A half-applied write is a machine state nobody asked
    for and nobody can reason about.

    Reads are served from cache: writables from the value retained at their
    last successful write, so get-after-set is bit-exact rather than a
    re-derivation; read-only variables from the last committed solve.

    The variable set is fixed at construction. Building it once is what lets
    :attr:`supported_variables` hand back the same object on every access,
    which callers may legitimately hold onto.

    This class never raises ``SystemExit``: whether an unusable model should
    end the process is the caller's decision, not this class's.

    **Not thread-safe, by design.** Every operation is synchronous and this
    class starts no threads, takes no locks, and does no I/O of its own. The
    atomicity above is transactional, not concurrent: it guarantees that one
    batch leaves no partial state behind, *not* that two batches may run at
    once. Two threads writing through the same model would interleave their
    snapshots and their lattice mutations, and the rollback would restore
    whichever snapshot it happened to hold — so the "complete no-op" promise
    does not survive concurrent use. The lattice underneath is a single
    mutable object shared by every variable, which is what makes this
    unavoidable rather than merely unimplemented. A caller that needs
    concurrent access must serialise it, one model per lattice, outside this
    class.
    """

    def __init__(
        self,
        *,
        simulator: PyATSimulator,
        action_variables: list[ActionVariable[PyATSimulator]],
    ) -> None:
        """Register the variables, seed the caches, and solve the boot orbit.

        The lattice is taken as built. Declared defaults are recorded as the
        retained input values but are **not** written to the lattice, so a
        caller handing over a lattice already in its nominal state keeps
        exactly that state, down to the last bit. A caller whose lattice
        differs from its declared defaults should :meth:`reset` afterwards to
        reconcile the two.

        Every binding is checked here, before anything is written: the element
        name must reach exactly one element, and a declared attribute must be
        one that element already has. Both are mistakes in the *definition* of
        a variable, so they belong at the point the variable set is adopted
        rather than at the first write that happens to touch it.

        Args:
            simulator: The simulator to drive. Its lattice is mutated in place.
            action_variables: The variables this model exposes. Each must bind
                an element the lattice actually has, and each writable must
                declare a float default value.

        Raises:
            UnknownElementError: a variable binds an element the lattice does
                not have. Raised before anything is written.
            AmbiguousElementError: a variable binds a name more than one
                element carries, so the binding addresses neither.
            AttributeError: a variable declares an attribute its element does
                not have — typically a typo. pyAT elements accept arbitrary
                attribute assignment, so left to the write path this would
                land on a dead field, be ignored by the solve, and read back
                intact: a silent no-op write.
            TypeError: a variable binds no lattice element at all.
            OrbitSolveError: the lattice as handed over has no stable closed
                orbit.
        """
        # Must exist before super().__init__, which registers every variable
        # through self.register_action_variable -- see that method.
        self._frozen = False

        super().__init__(simulator, action_variables)

        # Built once, and the same object on every access: the base property
        # returns a fresh dict per call, and get()/set() consult it per call.
        self._supported_variables: dict[str, ActionVariable[PyATSimulator]] = dict(
            self._action_variable_by_name
        )

        # Fail fast, before any write: a variable whose binding does not
        # resolve is a construction-time mistake, and finding it at the first
        # write instead would leave the caller guessing.
        for name, variable in self._supported_variables.items():
            self._validate_binding(name, variable)

        self._inputs: dict[str, float] = self._declared_defaults()
        simulator.solve()
        self._outputs: dict[str, float] = self._read_outputs()

        # Last statement: everything above runs while the model is still
        # mutable, including the base class's own registration calls.
        self._frozen = True

    # -- the lume-base surface --------------------------------------------

    @property
    def supported_variables(self) -> dict[str, ActionVariable[PyATSimulator]]:
        """Every variable this model exposes, keyed by name.

        The same object on every access — callers may hold onto it.
        """
        return self._supported_variables

    @property
    def lattice(self) -> at.Lattice:
        """The lattice this model drives, for callers that need pyAT directly."""
        return self.simulator.lattice

    def element_index(self, fam_name: str) -> int:
        """Index of the element named ``fam_name``.

        Raises:
            UnknownElementError: no element carries that name.
        """
        return self.simulator.element_index(fam_name)

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return one value per name, from cache.

        Raises:
            UnknownElementError: a name is not a variable of this model.
        """
        values: dict[str, Any] = {}
        for name in names:
            variable = self._require_variable(name)
            values[name] = (
                self._inputs[name]
                if isinstance(variable, WritableActionMixin)
                else self._outputs[name]
            )
        return values

    def _set(self, values: dict[str, Any]) -> None:
        """Apply a batch atomically and re-solve the closed orbit exactly once.

        Args:
            values: variable name -> value. Absolute, not a delta.

        Raises:
            UnknownElementError: a name is not a variable of this model, or is
                not one of its settable inputs.
            OrbitSolveError: the combined write leaves the lattice without a
                stable closed orbit. Every mutated element is restored, the
                simulator's cached solve is put back, and neither the retained
                inputs nor the cached outputs are touched, so a rejected write
                is a complete no-op.
        """
        # Validate before dispatching. The base _set silently skips anything
        # that is not writable, so a direct write to a read-only variable would
        # otherwise look like it succeeded; and its dispatch raises KeyError
        # for an unknown name, which tells a caller nothing.
        variables = [self._require_settable(name) for name in values]

        snapshots = self._snapshot(variables)
        # The solve is captured alongside the lattice because it can outlive a
        # failure: solve() may succeed and replace the simulator's cached
        # reading, only for _read_outputs to fail after it. Restoring the
        # lattice alone would then leave that cache describing an orbit the
        # lattice no longer has.
        solution = self.simulator.snapshot_solution()
        try:
            super()._set(values)
            self.simulator.solve()
            outputs = self._read_outputs()
        except Exception:
            self._restore(snapshots)
            self.simulator.restore_solution(solution)
            raise

        # Commit only now: a partially applied write must never be visible.
        self._inputs.update(values)
        self._outputs = outputs

    def reset(self) -> None:
        """Write every writable's default back, as one atomic batch.

        One batch means one solve and the same all-or-nothing guarantee as any
        other write. The lattice is never rebuilt, so construction-time
        misalignments and any other seeded state survive — this undoes writes,
        not faults.

        Raises:
            OrbitSolveError: the default configuration itself has no stable
                closed orbit, which can only happen when seeded faults alone
                destabilise the lattice. The lattice is left as it was.
        """
        self._set(self._declared_defaults())

    def register_action_variable(
        self, action_variable: ActionVariable[PyATSimulator]
    ) -> None:
        """Register a variable. Valid only during construction.

        Raises:
            NotImplementedError: the model is already built. The variable set
                is fixed so :attr:`supported_variables` and the value caches
                cannot drift out of step with it.
        """
        if self._frozen:
            raise NotImplementedError(
                "the variable set is fixed at construction; "
                "build a new model to expose different variables"
            )
        super().register_action_variable(action_variable)

    def unregister_action_variable(self, name: str) -> ActionVariable[PyATSimulator]:
        """Remove a variable. Valid only during construction.

        Raises:
            NotImplementedError: the model is already built.
        """
        if self._frozen:
            raise NotImplementedError(
                "the variable set is fixed at construction; "
                "build a new model to expose different variables"
            )
        return super().unregister_action_variable(name)

    # -- internals ---------------------------------------------------------

    def _validate_binding(self, name: str, variable: Variable) -> None:
        """Check one variable's bindings against the lattice. See __init__."""
        if isinstance(variable, PyATLatticeScalarVariable):
            # The lattice kind binds the ring, not an element. What can be
            # wrong in its definition is a snapshot target -- a subclass
            # declaring an attribute that does not exist -- and that is
            # checked here for the same reason an element attribute is.
            for index, attribute in variable.snapshot_targets(self.simulator):
                target = self.lattice if index is None else self.lattice[index]
                if not hasattr(target, attribute):
                    where = "the lattice" if index is None else f"element {index}"
                    raise AttributeError(
                        f"variable {name!r} declares attribute {attribute!r} "
                        f"as a snapshot target, which {where} does not have"
                    )
            return

        bindings = getattr(variable, "bindings", None)
        if bindings is not None:
            for binding in bindings:
                self._validate_element_binding(name, binding)
            return

        # Read-only variables name a monitor and no attribute; there is
        # nothing to check for them here beyond the name, and the boot solve
        # catches a non-monitor.
        element_name = getattr(variable, "element_name", None)
        if element_name is None:
            raise TypeError(
                f"variable {name!r} binds no lattice element; "
                "this model takes pyAT action variables"
            )
        self.simulator.unique_element_index(element_name)

    def _validate_element_binding(self, name: str, binding: ElementBinding) -> None:
        index = self.simulator.unique_element_index(binding.element_name)
        element = self.lattice[index]
        if not hasattr(element, binding.attribute):
            raise AttributeError(
                f"variable {name!r} declares attribute {binding.attribute!r}, "
                f"which element {binding.element_name!r} does not have"
            )
        if binding.index is not None:
            length = len(getattr(element, binding.attribute))
            if not -length <= binding.index < length:
                raise IndexError(
                    f"variable {name!r} declares index {binding.index} into "
                    f"{binding.attribute!r} of element {binding.element_name!r}, "
                    f"which has {length} entries"
                )

    def _declared_defaults(self) -> dict[str, float]:
        return {
            name: float(variable.default_value)
            for name, variable in self._supported_variables.items()
            if isinstance(variable, WritableActionMixin)
        }

    def _read_outputs(self) -> dict[str, float]:
        """One value per read-only variable, off the simulator's last solve."""
        return {
            name: variable._get(self.simulator)
            for name, variable in self._supported_variables.items()
            if not isinstance(variable, WritableActionMixin)
        }

    def _require_variable(self, name: str) -> Variable:
        variable = self._supported_variables.get(name)
        if variable is None:
            raise UnknownElementError(f"{name!r} is not a variable of this model")
        return variable

    def _require_settable(self, name: str) -> Variable:
        variable = self._require_variable(name)
        if not isinstance(variable, WritableActionMixin) or variable.read_only:
            raise UnknownElementError(f"{name!r} is not a settable input of this model")
        return variable

    def _snapshot(self, variables: list[Variable]) -> list[_Snapshot]:
        """Capture every attribute the batch will touch, as the variables declare.

        Each variable answers ``snapshot_targets`` with the ``(element index,
        attribute)`` pairs its write reaches -- one per binding for an element
        variable. Snapshots are per *attribute*, not per index, so an indexed
        write is covered by the whole sequence it writes into. Aliased names
        such as ``K`` need no special handling: pyAT routes them onto their
        underlying storage, so capturing and restoring the declared name
        restores exactly what the write changed.
        """
        snapshots: list[_Snapshot] = []
        seen: set[tuple[int | None, str]] = set()
        for variable in variables:
            for index, attribute in variable.snapshot_targets(self.simulator):
                key = (index, attribute)
                if key in seen:
                    continue
                seen.add(key)
                held = getattr(self._snapshot_target(index), attribute)
                snapshots.append((index, attribute, deepcopy(held)))
        return snapshots

    def _restore(self, snapshots: list[_Snapshot]) -> None:
        # Lattice-level targets first, then elements, each group in capture
        # order. pyAT's Lattice.energy setter pushes the value onto every
        # cavity, so restoring the ring last would overwrite a cavity that
        # had already been put back at its own snapshot.
        ordered = sorted(snapshots, key=lambda snapshot: snapshot[0] is not None)
        for index, attribute, value in ordered:
            setattr(self._snapshot_target(index), attribute, value)

    def _snapshot_target(self, index: int | None) -> Any:
        """The object a snapshot target names: an element, or the lattice."""
        return self.lattice if index is None else self.lattice[index]
