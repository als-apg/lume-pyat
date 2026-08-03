"""``PyATSimulator`` — a persistent, in-place pyAT lattice wrapper."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import at
import numpy as np

from lume_pyat.exceptions import (
    AmbiguousElementError,
    OrbitSolveError,
    UnknownElementError,
)
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

    Element lookup is by ``FamName``, so a name is only usable as an address
    if exactly one element carries it. Duplicates are legal in general — a
    lattice may hold any number of identically named drifts, and
    :meth:`element_index` resolves such a name to its last occurrence — but
    they are rejected wherever a name has to address something. Monitor names
    are checked here, at construction, because :meth:`solve` keys its readings
    by ``FamName`` and two monitors sharing one would silently return a single
    reading for both. Names bound by action variables are checked through
    :meth:`unique_element_index`.
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
            AmbiguousElementError: two monitors share a ``FamName``, which
                would make :meth:`solve` return one reading for both.
            UnknownElementError: a misalignment names an element the lattice
                does not have. Nothing has been mutated when this is raised.
        """
        self._lattice = lattice
        self._index_by_famname: dict[str, int] = {
            element.FamName: index for index, element in enumerate(lattice)
        }
        self._famname_counts = Counter(element.FamName for element in lattice)
        self._last_solution: dict[str, tuple[float, float]] | None = None

        # Before anything is mutated: a monitor that cannot be told apart from
        # another is not readable, and finding that out at the first solve --
        # as a reading quietly missing from the result -- would be far worse
        # than finding it out here.
        monitor_counts = Counter(
            element.FamName for element in lattice if isinstance(element, at.Monitor)
        )
        duplicated = sorted(name for name, count in monitor_counts.items() if count > 1)
        if duplicated:
            raise AmbiguousElementError(
                "monitor names must be unique for their readings to be "
                f"addressable; the lattice repeats {', '.join(map(repr, duplicated))}"
            )

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

    def unique_element_index(self, fam_name: str) -> int:
        """Index of the *sole* element named ``fam_name``.

        The lookup to use when a name is being adopted as an address — a
        variable binding, say — rather than resolved once. :meth:`element_index`
        answers "where does this name lead"; this answers "does this name lead
        somewhere unambiguous", and the two differ only for a repeated name.

        Raises:
            UnknownElementError: no element carries that name.
            AmbiguousElementError: more than one does.
        """
        index = self.element_index(fam_name)
        count = self._famname_counts[fam_name]
        if count > 1:
            raise AmbiguousElementError(
                f"{count} lattice elements are named {fam_name!r}; a name used "
                "to address an element must belong to exactly one"
            )
        return index

    def element(self, fam_name: str) -> at.Element:
        """The element named ``fam_name``, for direct attribute access.

        Raises:
            UnknownElementError: no element carries that name.
        """
        return self._lattice[self.element_index(fam_name)]

    def snapshot_solution(self) -> dict[str, tuple[float, float]] | None:
        """The cached solve result, or ``None`` when none has succeeded yet.

        The read half of a solve-rollback, and the reason it exists rather
        than callers using :attr:`last_solution`: this never raises, so a
        caller can capture and later restore "nothing solved yet" as
        faithfully as it restores a reading. Pair with
        :meth:`restore_solution`.
        """
        return self._last_solution

    def restore_solution(self, snapshot: dict[str, tuple[float, float]] | None) -> None:
        """Put a :meth:`snapshot_solution` result back as the cached solve.

        For a caller undoing a write it has already solved on: restoring the
        lattice alone would leave :attr:`last_solution` describing an orbit
        the lattice no longer has.
        """
        self._last_solution = snapshot

    def solve(self) -> dict[str, tuple[float, float]]:
        """Solve the closed orbit once and read it out at every monitor.

        On success :attr:`last_solution` is replaced with the new reading. On
        failure it is left alone: a failed solve says nothing about the orbit
        that was there before it, so the previous reading remains the best
        available answer rather than being discarded.

        A ring with no monitors solves to an empty reading rather than an
        error. The stability guards still run — an empty result means "nothing
        to read here", never "nothing was checked" — and a caller that does
        expect readings finds out at model construction instead. See
        :func:`~lume_pyat.solve.solve_orbit`.

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
