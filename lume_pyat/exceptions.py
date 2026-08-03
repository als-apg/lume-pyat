"""Exceptions raised by :mod:`lume_pyat`.

This module is deliberately standard-library only — not even ``numpy`` is
imported. Consumers re-export these classes to give their own callers a stable
type to catch, and that re-export must stay cheap enough to sit on an error
path without dragging in pyAT, lume-base, or HDF5. The guarantee is pinned by
a test.
"""

from __future__ import annotations


class OrbitSolveError(Exception):
    """Raised when a ring's one-turn map or closed orbit is unstable or non-finite.

    This is the load-bearing safety semantic of solving a nonlinear ring: an
    unstable or destabilized magnet configuration presents as
    NaN-without-exception — ``at.find_m44`` and ``at.find_orbit4`` emit an
    ``at.AtWarning`` and return non-finite garbage rather than raising. A
    caller that did not check would fail *open*, serving a garbage orbit to a
    monitor readback instead of refusing to.

    :func:`lume_pyat.solve.solve_orbit` detects instability by value and
    raises this instead. It covers three guard conditions, checked in order: a
    non-finite one-turn matrix, a transverse-plane one-turn trace with
    ``|trace| >= 2.0``, or a non-finite closed orbit.

    Derives from :class:`Exception` rather than :class:`ValueError`: an
    unstable lattice is a failure of the solve, not a bad argument.
    """


class UnknownElementError(ValueError):
    """Raised when a name does not resolve to an element of the lattice.

    Derives from :class:`ValueError` so callers that only care that a lookup
    was rejected can catch the broader type, and so consumers may alias this
    class into their own namespace without changing what an existing
    ``except ValueError`` catches.

    The message wording is part of the contract — downstream test suites match
    on it — so build the canonical message with :meth:`for_name` rather than
    formatting it at the raise site. The constructor stays unconstrained:
    callers that alias this class raise it for their own lookup failures with
    their own wording.
    """

    @classmethod
    def for_name(cls, name: str) -> UnknownElementError:
        """Build the error for a lattice-element lookup that found nothing."""
        return cls(f"no lattice element named {name!r}")


class AmbiguousElementError(UnknownElementError):
    """Raised when a name resolves to more than one element of the lattice.

    Element addressing is by ``FamName``, so a name carried by two elements
    picks out neither of them. Duplicates are only a problem for the names a
    caller actually addresses — a lattice may hold any number of identically
    named drifts — so this is raised where an addressable name is established,
    never merely because a name repeats.

    Derives from :class:`UnknownElementError` because it is the same failure
    from the caller's side: the name cannot be used to reach an element, and a
    consumer that already catches unresolvable names keeps catching this one.
    """
