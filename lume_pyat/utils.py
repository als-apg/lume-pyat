"""Element misalignment transforms, ported from pySC.

Ported from pySC (Python Simulated Commissioning,
https://github.com/kparasch/pySC): a magnet-misalignment transform built out
of a 3D rotation and the translation/R-matrix pair it implies. Every formula
here is a straight arithmetic/geometric port of the corresponding pySC
routine, not a reimplementation from first principles, so its sign and roll
conventions match pySC's — and, by construction, pyAT's, since pySC's
``update_transformation`` builds ``T1``/``T2``/``R1``/``R2`` directly.

Pure numpy: these operate on lattice elements by attribute access, without
importing ``at``.

Provenance: :func:`apply_misalignment` ports
``pySC.utils.sc_tools.update_transformation``, restricted to the dx/dy/roll
degrees of freedom (no dz/yaw/pitch); the three private helpers port
``sc_tools.rotation``, ``sc_tools._translation_vector``, and
``sc_tools._r_matrix``.
"""

from __future__ import annotations

import numpy as np


def _rotation_matrix_3d(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """3D extrinsic Z-Y-X rotation (Roll, Yaw, Pitch), ported from pySC's
    `sc_tools.rotation`."""
    ax, ay, az = pitch, yaw, roll
    return np.array(
        [
            [np.cos(ay) * np.cos(az), -np.cos(ay) * np.sin(az), np.sin(ay)],
            [
                np.cos(az) * np.sin(ax) * np.sin(ay) + np.cos(ax) * np.sin(az),
                np.cos(ax) * np.cos(az) - np.sin(ax) * np.sin(ay) * np.sin(az),
                -np.cos(ay) * np.sin(ax),
            ],
            [
                -np.cos(ax) * np.cos(az) * np.sin(ay) + np.sin(ax) * np.sin(az),
                np.cos(az) * np.sin(ax) + np.cos(ax) * np.sin(ay) * np.sin(az),
                np.cos(ax) * np.cos(ay),
            ],
        ]
    )


def _translation_vector(
    ld: float,
    r3d: np.ndarray,
    xaxis_xyz: np.ndarray,
    yaxis_xyz: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Ported from pySC's `sc_tools._translation_vector`."""
    t_d0 = np.array(
        [-np.dot(offsets, xaxis_xyz), 0.0, -np.dot(offsets, yaxis_xyz), 0.0, 0.0, 0.0]
    )
    t_0 = np.array(
        [
            ld * r3d[2, 0] / r3d[2, 2],
            r3d[2, 0],
            ld * r3d[2, 1] / r3d[2, 2],
            r3d[2, 1],
            0.0,
            ld / r3d[2, 2],
        ]
    )
    return t_0 + t_d0


def _r_matrix(ld: float, r3d: np.ndarray) -> np.ndarray:
    """Ported from pySC's `sc_tools._r_matrix`."""
    return np.array(
        [
            [
                r3d[1, 1] / r3d[2, 2],
                ld * r3d[1, 1] / r3d[2, 2] ** 2,
                -r3d[0, 1] / r3d[2, 2],
                -ld * r3d[0, 1] / r3d[2, 2] ** 2,
                0.0,
                0.0,
            ],
            [0.0, r3d[0, 0], 0.0, r3d[1, 0], r3d[2, 0], 0.0],
            [
                -r3d[1, 0] / r3d[2, 2],
                -ld * r3d[1, 0] / r3d[2, 2] ** 2,
                r3d[0, 0] / r3d[2, 2],
                ld * r3d[0, 0] / r3d[2, 2] ** 2,
                0.0,
                0.0,
            ],
            [0.0, r3d[0, 1], 0.0, r3d[1, 1], r3d[2, 1], 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [
                -r3d[0, 2] / r3d[2, 2],
                -ld * r3d[0, 2] / r3d[2, 2] ** 2,
                -r3d[1, 2] / r3d[2, 2],
                -ld * r3d[1, 2] / r3d[2, 2] ** 2,
                0.0,
                1.0,
            ],
        ]
    )


def apply_misalignment(
    element, *, dx: float = 0.0, dy: float = 0.0, roll: float = 0.0
) -> None:
    """Set an AT element's `T1`/`T2`/`R1`/`R2` for a dx/dy/roll misalignment.

    Ports pySC's `sc_tools.update_transformation`, restricted to the dx/dy/roll
    degrees of freedom (dz = yaw = pitch = 0.0 always) -- longitudinal shift
    and out-of-plane tilt are out of scope. Bend-aware: for an element with a
    nonzero `BendingAngle`, the exit transform (`T2`/`R2`) accounts for the
    magnet's curved geometry via its `Length` and `BendingAngle`, matching
    pySC exactly for a straight element (`BendingAngle == 0`).

    ABSOLUTE, not additive: each call fully replaces any prior T1/T2/R1/R2
    misalignment on `element` -- it does not compose with a previous call.
    Pass the complete desired dx/dy/roll together in one call; calling this
    twice with, say, dx then dy separately does not accumulate both -- the
    second call overwrites the first's transform entirely.

    Args:
        element: An `at` lattice element (mutated in place).
        dx: Horizontal offset, in meters.
        dy: Vertical offset, in meters.
        roll: Roll about the beam axis, in radians, following pySC's
            `sc_tools.rotation` z-axis convention.

    Raises:
        ZeroDivisionError: never for a valid AT element -- pySC's own formulas
            divide by `r3d[2, 2]`, which is 1.0 for any dx/dy/roll-only
            (dz = yaw = pitch = 0) misalignment.
    """
    mag_length = getattr(element, "Length", 0.0)
    mag_theta = getattr(element, "BendingAngle", 0.0)

    offsets = np.array([dx, dy, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # Entrance transform.
    r_3d = _rotation_matrix_3d(0.0, 0.0, roll)
    ld = np.dot(np.dot(r_3d, z_axis), offsets)

    t_entrance = _translation_vector(
        ld, r_3d, np.dot(r_3d, x_axis), np.dot(r_3d, y_axis), offsets
    )
    element.R1 = _r_matrix(ld, r_3d)
    element.T1 = np.dot(np.linalg.inv(element.R1), t_entrance)

    # Exit transform -- bend-aware: undoes the entrance rotation in the frame
    # of the magnet's own curvature (RB), so a straight element (mag_theta ==
    # 0) reduces to the mirror image of the entrance transform.
    rx = r_3d
    rb = _rotation_matrix_3d(0.0, -mag_theta, 0.0)
    r_3d_exit = np.dot(rb.T, np.dot(rx.T, rb))
    op_p = np.array(
        [
            mag_length * (np.cos(mag_theta) - 1.0) / mag_theta if mag_theta else 0.0,
            0.0,
            mag_length * (np.sin(mag_theta) / mag_theta if mag_theta else 1.0),
        ]
    )
    op_p_prime = op_p - np.dot(rx, op_p) - offsets
    ld_exit = np.dot(np.dot(rb, z_axis), op_p_prime)

    element.T2 = _translation_vector(
        ld_exit, r_3d_exit, np.dot(rb, x_axis), np.dot(rb, y_axis), op_p_prime
    )
    element.R2 = _r_matrix(ld_exit, r_3d_exit)
