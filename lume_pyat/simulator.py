"""``PyATSimulator`` — a persistent, in-place pyAT lattice wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import at
import numpy as np

from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.solve import monitor_xy, solve_orbit
from lume_pyat.utils import apply_misalignment

__all__ = [
    "ElementState",
    "PyATSimulator",
    "restore_element",
    "snapshot_element",
]


@dataclass(frozen=True)
class ElementState:
    """Snapshot of one ring element's mutable strength fields.

    ``polynom_b`` and ``kick_angle`` are ``None`` when the element doesn't
    carry that field (correctors have no ``PolynomB``; magnets have no
    ``KickAngle``).
    """

    polynom_b: list[float] | None
    kick_angle: list[float] | None


def snapshot_element(element: at.Element) -> ElementState:
    """Capture ``element``'s current ``PolynomB``/``KickAngle`` state.

    For later write-rollback: pair with :func:`restore_element`.
    """
    return ElementState(
        polynom_b=list(element.PolynomB) if hasattr(element, "PolynomB") else None,
        kick_angle=list(element.KickAngle) if hasattr(element, "KickAngle") else None,
    )


def restore_element(element: at.Element, state: ElementState) -> None:
    """Write a previously captured :class:`ElementState` back onto ``element``."""
    if state.polynom_b is not None:
        element.PolynomB = list(state.polynom_b)
    if state.kick_angle is not None:
        element.KickAngle = list(state.kick_angle)


class PyATSimulator:
    """One persistent ``at.Lattice``, mutated in place.

    The caller builds the lattice and hands it over; this class never copies
    or rebuilds it. That is a deliberate deviation from the deepcopy-per-reset
    convention used elsewhere in the ecosystem, and it is what makes
    sequential writes compose the way their physical counterparts do: writing
    a device twice is idempotent (last value wins, not cumulative), and
    writing two independent devices in either order reaches the same final
    state. It also means construction-time misalignments survive every
    subsequent write, because there is no rebuild to lose them.

    Because the lattice is shared rather than copied, a caller that keeps its
    own reference sees the same mutations. That is intended — it is how a
    caller applies its own strength changes before calling :meth:`solve`.

    Element lookup is by ``FamName``. A lattice with duplicate names resolves
    each to its last occurrence, and :meth:`solve` likewise returns one entry
    per distinct monitor name, so monitor names should be unique if every
    monitor is to be readable.
    """

    def __init__(
        self,
        lattice: at.Lattice,
        *,
        element_misalignments: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Adopt ``lattice`` and seed any misalignments onto it.

        Args:
            lattice: The lattice to own. Mutated in place, never copied.
            element_misalignments: ``FamName`` -> keyword arguments for
                :func:`~lume_pyat.utils.apply_misalignment` (``dx``/``dy``/
                ``roll``, all optional). Applied once, here; an element absent
                from this mapping keeps its existing transforms. Every name is
                validated against the lattice **before** any element is
                mutated, because the transform is absolute rather than
                additive — a partial application could not be undone by
                retrying with corrected arguments.

        Raises:
            UnknownElementError: a misalignment names an element the lattice
                does not have. Nothing has been mutated when this is raised.
        """
        self._lattice = lattice
        self._index_by_famname: dict[str, int] = {
            element.FamName: index for index, element in enumerate(lattice)
        }
        self._last_solution: dict[str, tuple[float, float]] | None = None

        misalignments = element_misalignments or {}
        for fam_name in misalignments:
            self.element_index(fam_name)  # validate every name before mutating anything
        for fam_name, misalignment in misalignments.items():
            apply_misalignment(self.element(fam_name), **misalignment)

    @property
    def lattice(self) -> at.Lattice:
        """The lattice this simulator owns, for callers that need pyAT directly."""
        return self._lattice

    @property
    def last_solution(self) -> dict[str, tuple[float, float]]:
        """The most recent *successful* :meth:`solve` result.

        Reading this is O(1): consumers that need many values from one solve
        read them from here rather than re-solving per value.

        Raises:
            OrbitSolveError: no solve has succeeded yet.
        """
        if self._last_solution is None:
            raise OrbitSolveError(
                "no closed orbit has been solved yet -- call solve() first"
            )
        return self._last_solution

    def element_index(self, fam_name: str) -> int:
        """Index of the element named ``fam_name``.

        Raises:
            UnknownElementError: no element carries that name.
        """
        index = self._index_by_famname.get(fam_name)
        if index is None:
            raise UnknownElementError.for_name(fam_name)
        return index

    def element(self, fam_name: str) -> at.Element:
        """The element named ``fam_name``, for direct attribute access.

        Raises:
            UnknownElementError: no element carries that name.
        """
        return self._lattice[self.element_index(fam_name)]

    def solve(self) -> dict[str, tuple[float, float]]:
        """Solve the closed orbit once and read it out at every monitor.

        On success :attr:`last_solution` is replaced with the new reading. On
        failure it is left alone: a failed solve says nothing about the orbit
        that was there before it, so the previous reading remains the best
        available answer rather than being discarded.

        Returns:
            ``FamName`` -> ``(x, y)`` in meters, one entry per monitor.

        Raises:
            OrbitSolveError: the lattice has no trustworthy closed orbit.
        """
        # solve_orbit already guards the usual failure, which pyAT signals by
        # value (NaN). In rarer configurations pyAT raises instead: at.AtError
        # on a failure it detects itself, or a bare LinAlgError/ValueError out
        # of LAPACK or pyAT internals. All three are the same failure from
        # here -- no orbit can be trusted -- so they are folded into
        # OrbitSolveError and a caller needs only one except clause.
        #
        # The broad except stays scoped to this one call on purpose:
        # UnknownElementError is itself a ValueError, and a wider catch would
        # relabel "unknown element" as "unstable orbit".
        try:
            orbit_at_monitors = solve_orbit(self._lattice)
        except (at.AtError, np.linalg.LinAlgError, ValueError) as exc:
            raise OrbitSolveError(
                f"closed orbit solve raised {type(exc).__name__}: {exc}"
            ) from exc

        solution = {
            fam_name: (x, y)
            for fam_name, x, y in monitor_xy(self._lattice, orbit_at_monitors)
        }
        self._last_solution = solution
        return solution
