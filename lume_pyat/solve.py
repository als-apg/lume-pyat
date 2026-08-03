"""Guarded 4D closed-orbit solve for a pyAT ring.

This is the load-bearing safety semantic of solving a nonlinear ring: an
unstable or destabilized magnet configuration presents as
NaN-without-exception -- ``at.find_m44``/``at.find_orbit4`` emit an
``at.AtWarning`` and return non-finite garbage rather than raising. A caller
that just checked ``np.isfinite`` on a bare ``find_orbit4`` call, or worse,
didn't check at all, would fail *open*: it would serve a garbage orbit as a
monitor readback instead of refusing to.

:func:`solve_orbit` closes that gap by detecting instability *by value*
(non-finite one-turn-matrix entries, |trace| >= 2.0 in either transverse
plane, or a non-finite closed orbit) and raising :class:`OrbitSolveError`
instead. pyAT's own warnings are expected noise on the failure path here --
they are suppressed inside this module so a guard trip surfaces to the
caller as a single exception, not console spam plus an exception.

Ring-facing: this module only imports ``at``/``numpy`` (plus the stdlib and
the package's own exceptions), so it stays valid for any 4D-canonical ring,
independent of how that ring gets served.
"""

from __future__ import annotations

import warnings

import at
import numpy as np

from lume_pyat.exceptions import OrbitSolveError

# |trace| at or above this value marks a transverse plane's one-turn map as
# unstable (a stable linear map has both eigenvalues on the unit circle,
# which bounds |trace| = |lambda + 1/lambda| < 2).
_TRACE_INSTABILITY_THRESHOLD = 2.0


def _monitor_refpts(ring: at.Lattice) -> np.ndarray:
    """Indices of every `at.Monitor` element in `ring`, selected by type.

    A ring carrying no `at.Monitor` yields an empty array, and the functions
    below then yield an empty result rather than raising -- see
    :func:`solve_orbit` for why that is the contract.
    """
    return np.array(
        [i for i, element in enumerate(ring) if isinstance(element, at.Monitor)]
    )


def monitor_xy(
    ring: at.Lattice, orbit_at_monitors: np.ndarray
) -> list[tuple[str, float, float]]:
    """Per-monitor ``(FamName, x, y)`` readout of a :func:`solve_orbit` result.

    The single shared "read the solved orbit at the monitors" primitive.
    Callers that need monitor readings should key them off this rather than
    re-deriving the selection, which is what keeps their row -> element
    alignment identical by construction -- `orbit_at_monitors` rows are
    ordered by the same :func:`_monitor_refpts` selection
    :func:`solve_orbit` solved at.

    Args:
        ring: The lattice `orbit_at_monitors` was solved on.
        orbit_at_monitors: A :func:`solve_orbit` result for `ring`, shape
            `(n_monitors, 6)`.

    Returns:
        One ``(FamName, x_m, y_m)`` tuple per `at.Monitor` element, in ring
        order (e.g. ``("BPM01", 1.2e-6, -3.4e-6)``), with x/y taken from
        orbit coordinates 0 and 2.
    """
    return [
        (
            ring[el_idx].FamName,
            float(orbit_at_monitors[row, 0]),
            float(orbit_at_monitors[row, 2]),
        )
        for row, el_idx in enumerate(_monitor_refpts(ring))
    ]


def solve_orbit(ring: at.Lattice) -> np.ndarray:
    """Guarded 4D closed-orbit solve at every `at.Monitor` refpt in `ring`.

    Runs `at.find_m44` and `at.find_orbit4` with pyAT's `AtWarning`
    instability warnings suppressed (they are the expected shape of the
    failure this function guards against, not information the caller
    needs), then checks each result by value before trusting it:

    1. `find_m44`'s one-turn matrix must be entirely finite.
    2. Its transverse-plane traces (x-block `m44[0,0] + m44[1,1]`,
       y-block `m44[2,2] + m44[3,3]`) must both satisfy `|trace| < 2.0`.
    3. `find_orbit4`'s closed orbit at the monitor refpts must be entirely
       finite.

    A ring with no `at.Monitor` elements is not an error here: the guards
    still run against the one-turn matrix, and the result is an empty
    `(0, 6)` array. That is deliberate. Whether a monitorless ring is a
    mistake depends on what the caller wants -- driving magnets and reading
    setpoints back needs no monitors at all -- so this function reports the
    orbit at the monitors that exist rather than legislating how many there
    should be. A caller that *does* expect readings finds out precisely:
    binding a read-only variable to an element that is not a monitor raises
    `UnknownElementError` naming that element, at model construction.

    Args:
        ring: A 4D-canonical `at.Lattice` (radiation/cavity disabled, e.g.
            via `ring.disable_6d()`).

    Returns:
        The `find_orbit4` closed-orbit array at every `at.Monitor` refpt,
        shape `(n_monitors, 6)`.

    Raises:
        OrbitSolveError: if any of the three guard conditions above trips --
            the ring configuration is unstable and no orbit can be trusted.
    """
    refpts = _monitor_refpts(ring)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        m44 = at.find_m44(ring)[0]

    if not np.all(np.isfinite(m44)):
        raise OrbitSolveError("find_m44 one-turn matrix has non-finite entries")

    trace_x = m44[0, 0] + m44[1, 1]
    trace_y = m44[2, 2] + m44[3, 3]
    if (
        abs(trace_x) >= _TRACE_INSTABILITY_THRESHOLD
        or abs(trace_y) >= _TRACE_INSTABILITY_THRESHOLD
    ):
        raise OrbitSolveError(
            f"one-turn matrix unstable: |trace_x| = {abs(trace_x):.3f}, "
            f"|trace_y| = {abs(trace_y):.3f} (threshold {_TRACE_INSTABILITY_THRESHOLD})"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        _, orbit_at_monitors = at.find_orbit4(ring, refpts=refpts)

    if not np.all(np.isfinite(orbit_at_monitors)):
        raise OrbitSolveError("find_orbit4 returned a non-finite closed orbit")

    return np.asarray(orbit_at_monitors)
