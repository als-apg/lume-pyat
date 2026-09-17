"""Guarded closed-orbit solve for a pyAT ring, in 4D or 6D.

This is the load-bearing safety semantic of solving a nonlinear ring: an
unstable or destabilized magnet configuration presents as
NaN-without-exception -- ``at.find_m66``/``at.find_orbit4``/``at.find_orbit6``
emit an ``at.AtWarning`` and return non-finite garbage rather than raising,
or, with a cavity in the ring, return a perfectly finite one-turn matrix whose
eigenvalues have left the unit circle. A caller that just checked
``np.isfinite`` on a bare orbit call, or worse, didn't check at all, would
fail *open*: it would serve a garbage orbit as a monitor readback instead of
refusing to.

:func:`solve_orbit` closes that gap by detecting instability *by value*
(non-finite one-turn-matrix entries, a one-turn eigenvalue outside the unit
circle in any of the three planes, or a non-finite closed orbit) and raising
:class:`OrbitSolveError` instead. pyAT's own warnings are expected noise on
the failure path here -- they are suppressed inside this module so a guard
trip surfaces to the caller as a single exception, not console spam plus an
exception.

Ring-facing: this module only imports ``at``/``numpy`` (plus the stdlib and
the package's own exceptions), so it stays valid for any ``at.Lattice``,
independent of how that ring gets served.
"""

from __future__ import annotations

import warnings

import at
import numpy as np

from lume_pyat.exceptions import OrbitSolveError

# A one-turn eigenvalue with |eigenvalue| above 1 + this marks its plane as
# unstable. A stable symplectic map has every eigenvalue on the unit circle;
# the tolerance absorbs round-off in the eigen-decomposition (a cavity-less
# ring's longitudinal block is a Jordan block, whose eigenvalues are the
# least accurately computed) without letting a real excursion through.
_EIGENVALUE_TOLERANCE = 1.0e-6

_PLANES = ("horizontal", "vertical", "longitudinal")


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


def _eigenvalue_magnitude_per_plane(m66: np.ndarray) -> list[float]:
    """Largest |eigenvalue| of the one-turn map, attributed to each plane.

    The stability decision is made on the full 6x6 spectrum, which is right
    for a coupled ring where no 2x2 block tells the whole story. The plane
    labels are for the error message: each eigenvalue is attributed to the
    plane whose uncoupled 2x2 diagonal block has the nearest eigenvalue, two
    per plane. For an uncoupled ring that is exact; for a coupled one it names
    the plane the mode mostly lives in.
    """
    values = np.linalg.eigvals(m66)
    block_values = [
        np.linalg.eigvals(m66[2 * plane : 2 * plane + 2, 2 * plane : 2 * plane + 2])
        for plane in range(3)
    ]
    distances = sorted(
        (float(np.min(np.abs(value - block_values[plane]))), k, plane)
        for k, value in enumerate(values)
        for plane in range(3)
    )

    assigned: dict[int, int] = {}
    per_plane_count = [0, 0, 0]
    for _, k, plane in distances:
        if k in assigned or per_plane_count[plane] == 2:
            continue
        assigned[k] = plane
        per_plane_count[plane] += 1

    worst = [0.0, 0.0, 0.0]
    for k, plane in assigned.items():
        worst[plane] = max(worst[plane], float(abs(values[k])))
    return worst


def solve_orbit(ring: at.Lattice) -> np.ndarray:
    """Guarded closed-orbit solve at every `at.Monitor` refpt in `ring`.

    Runs `at.find_m66` and then the orbit solver matching the ring's
    dimensionality -- `at.find_orbit6` for a 6D ring (``ring.is_6d``, i.e. a
    cavity is enabled), `at.find_orbit4` for a 4D one -- with pyAT's
    `AtWarning` instability warnings suppressed (they are the expected shape
    of the failure this function guards against, not information the caller
    needs), then checks each result by value before trusting it:

    1. `find_m66`'s one-turn matrix must be entirely finite.
    2. Every eigenvalue of that matrix must satisfy
       `|eigenvalue| <= 1 + 1e-6`, in all three planes. This is what catches
       a ring with a cavity: its one-turn matrix stays finite while an
       eigenvalue leaves the unit circle. The error names the unstable
       plane(s).
    3. The closed orbit at the monitor refpts must be entirely finite.

    The dispatch on ``ring.is_6d`` is not optional: pyAT refuses to solve a
    cavity-less ring in 6D and a ring with an enabled cavity in 4D.

    Rings with radiation enabled are out of scope. The guards would still
    run -- a damped map has its eigenvalues strictly inside the unit circle,
    which passes -- but nothing here accounts for the energy loss, and the
    orbit such a ring solves to is not the one this package's variables
    describe.

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
        ring: An `at.Lattice`, either 4D (no enabled cavity, e.g. via
            `ring.disable_6d()`) or 6D with the cavity enabled and radiation
            off (`ring.enable_6d(at.RFCavity)`).

    Returns:
        The closed-orbit array at every `at.Monitor` refpt, shape
        `(n_monitors, 6)`.

    Raises:
        OrbitSolveError: if any of the three guard conditions above trips --
            the ring configuration is unstable and no orbit can be trusted.
            pyAT's own `at.AtError` (for instance when no synchronous phase
            can be found for a 6D ring) passes through unchanged;
            :class:`~lume_pyat.simulator.PyATSimulator` folds it.
    """
    refpts = _monitor_refpts(ring)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        m66 = at.find_m66(ring)[0]

    if not np.all(np.isfinite(m66)):
        raise OrbitSolveError("find_m66 one-turn matrix has non-finite entries")

    worst = _eigenvalue_magnitude_per_plane(m66)
    unstable = [
        plane
        for plane, magnitude in zip(_PLANES, worst, strict=True)
        if magnitude > 1.0 + _EIGENVALUE_TOLERANCE
    ]
    if unstable:
        where = (
            f"{unstable[0]} plane"
            if len(unstable) == 1
            else f"{', '.join(unstable[:-1])} and {unstable[-1]} planes"
        )
        per_plane = ", ".join(
            f"{plane} {magnitude:.3e}"
            for plane, magnitude in zip(_PLANES, worst, strict=True)
        )
        raise OrbitSolveError(
            f"one-turn map unstable in the {where}: |eigenvalue| = "
            f"{max(worst):.3e} (limit 1 + {_EIGENVALUE_TOLERANCE:.0e}); "
            f"per plane: {per_plane}"
        )

    find_orbit = at.find_orbit6 if ring.is_6d else at.find_orbit4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        _, orbit_at_monitors = find_orbit(ring, refpts=refpts)

    if not np.all(np.isfinite(orbit_at_monitors)):
        raise OrbitSolveError(
            f"{find_orbit.__name__} returned a non-finite closed orbit"
        )

    return np.asarray(orbit_at_monitors)
