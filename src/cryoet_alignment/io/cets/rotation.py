"""Rotation operators of the profile — the standard right-handed active matrices of RELION's
``t3Matrix::rotation`` port (``io.relion.alignment``), so the codec cannot drift from the pinned convention.

``R = Rz(rot) · Ry(tilt) · Rx(xrot)`` maps centred tomogram coordinates to centred projection coordinates
(then drop z): ``rot`` = tilt-axis rotation (AreTomo3 ROT / RELION rlnTomoZRot / Warp AxisAngle),
``tilt`` = refined tilt (TILT / rlnTomoYTilt / -(Angle+LevelAngleY)), ``xrot`` = rlnTomoXTilt / LevelAngleX.
"""

import math
from typing import Tuple

import numpy as np

from cryoet_alignment.io.relion.alignment import rot_x, rot_y, rot_z

__all__ = ["rot_x", "rot_y", "rot_z", "rotation", "tilt_matrix", "in_plane_matrix", "decompose", "check_rotation"]


def rotation(rot_deg: float, tilt_deg: float, xrot_deg: float) -> np.ndarray:
    return rot_z(rot_deg) @ rot_y(tilt_deg) @ rot_x(xrot_deg)


def tilt_matrix(tilt_deg: float, xrot_deg: float) -> np.ndarray:
    """The ``tilt`` affine of the profile: ``Ry(tilt) · Rx(xrot)`` (3×3)."""
    return rot_y(tilt_deg) @ rot_x(xrot_deg)


def in_plane_matrix(rot_deg: float) -> np.ndarray:
    """The ``in_plane_rotation`` affine: ``Rz(rot)`` (3×3, acts on x/y, passes z)."""
    return rot_z(rot_deg)


def check_rotation(r: np.ndarray, what: str = "matrix", tol: float = 1e-6) -> None:
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"{what}: expected a 3x3 matrix, got shape {r.shape}")
    if not np.allclose(r.T @ r, np.eye(3), atol=tol):
        raise ValueError(f"{what}: not orthonormal (max |R^T R - I| = {np.abs(r.T @ r - np.eye(3)).max():.2e})")
    if abs(np.linalg.det(r) - 1.0) > tol:
        raise ValueError(f"{what}: det = {np.linalg.det(r):.6f}, expected +1 (no reflections)")


def decompose(r: np.ndarray, tol: float = 1e-9) -> Tuple[float, float, float]:
    """Recover ``(rot, tilt, xrot)`` in degrees from ``R = Rz(rot) Ry(tilt) Rx(xrot)``.

    ``R[2,0] = -sin(tilt)``, ``R[2,1:] = cos(tilt)·(sin xrot, cos xrot)``, ``R[:2,0] = cos(tilt)·(cos rot, sin rot)``.
    At ``|cos tilt| < 1e-12`` (gimbal lock, tilt = ±90°) ``xrot`` is set to 0 and ``rot`` absorbs the rest.
    The result is verified by rebuilding the matrix (``tol``).
    """
    r = np.asarray(r, dtype=np.float64)
    check_rotation(r, "projection rotation")
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    if abs(math.cos(math.radians(tilt))) < 1e-12:
        xrot = 0.0
        rot = math.degrees(math.atan2(-r[0, 1], r[1, 1]))
    else:
        xrot = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        rot = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    rebuilt = rotation(rot, tilt, xrot)
    err = float(np.abs(rebuilt - r).max())
    if err > tol:
        raise ValueError(f"rotation decomposition residual {err:.2e} exceeds {tol:.1e}")
    return rot, tilt, xrot
