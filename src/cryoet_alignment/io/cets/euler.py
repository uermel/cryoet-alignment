"""Particle orientation ⇄ ZYZ Euler triplets, transcribed from the reference implementations.

``zyz_to_matrix(rot, tilt, psi)`` is RELION's ``Euler::anglesToMatrix3`` (``src/jaz/math/Euler_angles_relion.h:38-48``),
which is entry for entry Warp's ``Matrix3.Euler`` (``WarpLib/Tools/Matrix3.cs:231-259``). The returned matrix ``A``
maps particle (reference map) coordinates to tomogram coordinates: RELION uses it un-transposed in
``ParticleSet::getMatrix4x4`` ("This maps coordinates from particle space to tomogram space",
``src/jaz/tomography/particle_set.cpp:428-457``) and Warp as ``ParticleMatrix`` in ``TiltSeries.cs:732-741``.
In terms of the active right-handed matrices of ``io.relion.alignment``: ``A = (Rz(rot) · Ry(tilt) · Rz(psi))ᵀ``.

``matrix_to_zyz(A)`` is Warp's ``Matrix3.EulerFromMatrix`` (``Matrix3.cs:289-320``) with its gimbal-lock branches.

The cryoET Data Portal stores ``Rotation.from_euler("ZYZ", (rot, tilt, psi)).inv()`` for RELION-sourced oriented
points (``common/point_converter.py:350,421``); scipy's intrinsic ZYZ is ``Aᵀ``, so the portal matrix *is* ``A``.
"""

import math
from typing import Tuple

import numpy as np

from cryoet_alignment.io.cets.rotation import check_rotation

__all__ = ["zyz_to_matrix", "matrix_to_zyz", "zyz_to_matrices", "matrices_to_zyz"]

_EPS = 16 * 1.192092896e-07  # Warp's float32 epsilon guard (Matrix3.cs:295)
_EPS_SIN = 1.192092896e-07


def zyz_to_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """RELION/Warp ``A(rot, tilt, psi)``: particle → tomogram (degrees in, 3×3 float64 out)."""
    p, t, c = (math.radians(float(v)) for v in (rot_deg, tilt_deg, psi_deg))
    sp, cp = math.sin(p), math.cos(p)
    st, ct = math.sin(t), math.cos(t)
    sc, cc = math.sin(c), math.cos(c)
    return np.array(
        [
            [cc * ct * cp - sc * sp, cc * ct * sp + sc * cp, -cc * st],
            [-sc * ct * cp - cc * sp, -sc * ct * sp + cc * cp, sc * st],
            [st * cp, st * sp, ct],
        ],
        dtype=np.float64,
    )


def matrix_to_zyz(a: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`zyz_to_matrix` (Warp ``EulerFromMatrix``); returns ``(rot, tilt, psi)`` in degrees.

    The result is verified by rebuilding the matrix (1e-9); at gimbal lock (``tilt`` = 0 or 180°) ``rot`` is 0
    and ``psi`` carries the in-plane rotation, exactly as Warp does.
    """
    a = np.asarray(a, dtype=np.float64)
    check_rotation(a, "particle rotation")
    m11, m13, m21, m23, m31, m32, m33 = a[0, 0], a[0, 2], a[1, 0], a[1, 2], a[2, 0], a[2, 1], a[2, 2]
    abs_sb = math.sqrt(m13 * m13 + m23 * m23)
    if abs_sb > _EPS:
        gamma = math.atan2(m23, -m13)
        alpha = math.atan2(m32, m31)
        if abs(math.sin(gamma)) < _EPS_SIN:
            sign_sb = math.copysign(1.0, -m13 / math.cos(gamma))
        else:
            sign_sb = math.copysign(1.0, m23) if math.sin(gamma) > 0 else -math.copysign(1.0, m23)
        beta = math.atan2(sign_sb * abs_sb, m33)
    elif m33 > 0:
        alpha, beta, gamma = 0.0, 0.0, math.atan2(-m21, m11)
    else:
        alpha, beta, gamma = 0.0, math.pi, math.atan2(m21, -m11)
    rot, tilt, psi = (math.degrees(v) for v in (alpha, beta, gamma))
    err = float(np.abs(zyz_to_matrix(rot, tilt, psi) - a).max())
    if err > 1e-9:
        raise ValueError(f"matrix_to_zyz: rebuilt matrix deviates by {err:.2e}")
    return rot, tilt, psi


def zyz_to_matrices(eulers_deg) -> np.ndarray:
    """(N,3) ``rot, tilt, psi`` degrees → (N,3,3)."""
    e = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    return np.stack([zyz_to_matrix(*row) for row in e]) if len(e) else np.zeros((0, 3, 3))


def matrices_to_zyz(matrices) -> np.ndarray:
    """(N,3,3) → (N,3) ``rot, tilt, psi`` degrees."""
    m = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    return np.array([matrix_to_zyz(x) for x in m], dtype=np.float64).reshape(-1, 3)
