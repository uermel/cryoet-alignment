"""CTF metadata in the profile: ``CTFMetadata{defocus_u, defocus_v, defocus_angle, phase_shift, defocus_handedness}``.

Units (profile convention until the standard specifies them): defocus in Å, underfocus positive,
``defocus_u >= defocus_v``; ``defocus_angle`` in DEGREES in ``[0, 180)`` measured from +x toward +y (the
direction of the larger defocus); ``phase_shift`` in DEGREES. Sources: AreTomo3 ``_CTF.txt`` and the portal
report the phase in radians (``CSaveCtfResults.cpp:93``); Warp stores ``Defocus ± DefocusDelta/2`` in µm and
the phase in multiples of π (``CTF.cs:194-197,461-512``); RELION uses Å and degrees. ``defocus_handedness``
has no agreed meaning in CETS yet: unknown handedness is written as an explicit ``null`` (the model would
otherwise materialise its ``-1`` default), and a value is only carried through when the source has one.
"""

import math
from typing import Optional, Tuple

from cryoet_alignment.io.cets.profile import require_cets


def canonical_astigmatism(defocus_u_a: float, defocus_v_a: float, angle_deg: float) -> Tuple[float, float, float]:
    """Enforce ``u >= v`` (swap and rotate the angle by 90°) and wrap the angle into ``[0, 180)``; both are
    exact identities of the ``cos(2(theta - angle))`` defocus field."""
    u, v, a = float(defocus_u_a), float(defocus_v_a), float(angle_deg)
    if v > u:
        u, v, a = v, u, a + 90.0
    a = a % 180.0
    return u, v, a


def ctf_metadata(
    *,
    defocus_u_a: float,
    defocus_v_a: float,
    defocus_angle_deg: float,
    phase_shift_deg: Optional[float] = None,
    defocus_handedness: Optional[int] = None,
):
    """Build a profile ``CTFMetadata`` (explicit ``None`` handedness, canonical astigmatism)."""
    m = require_cets()
    u, v, a = canonical_astigmatism(defocus_u_a, defocus_v_a, defocus_angle_deg)
    return m.CTFMetadata(
        defocus_u=u,
        defocus_v=v,
        defocus_angle=a,
        phase_shift=None if phase_shift_deg is None else float(phase_shift_deg),
        defocus_handedness=defocus_handedness,
    )


# ------------------------------------------------------------------ AreTomo3 _CTF.txt (Å, radians)


def from_aretomo3_row(row, defocus_handedness: Optional[int] = None):
    """``CtfInfo`` (DfMax/DfMin Å, azimuth degrees, phase radians) -> ``CTFMetadata``."""
    return ctf_metadata(
        defocus_u_a=row.df_max_a,
        defocus_v_a=row.df_min_a,
        defocus_angle_deg=row.azimuth_deg,
        phase_shift_deg=math.degrees(row.phase_rad),
        defocus_handedness=defocus_handedness,
    )


def to_aretomo3_row(ctf, micrograph: int, df_hand: Optional[int] = None):
    """``CTFMetadata`` -> ``CtfInfo``. Score / resolution are not part of CETS: documented placeholders
    ``0.0`` / ``999.99`` are written (never read back as measurements)."""
    from cryoet_alignment.io.aretomo3.ctf import CtfInfo

    if ctf.defocus_u is None or ctf.defocus_v is None or ctf.defocus_angle is None:
        raise ValueError("CTFMetadata is incomplete (defocus_u/defocus_v/defocus_angle required for _CTF.txt)")
    return CtfInfo(
        micrograph=micrograph,
        df_max_a=float(ctf.defocus_u),
        df_min_a=float(ctf.defocus_v),
        azimuth_deg=float(ctf.defocus_angle),
        phase_rad=math.radians(float(ctf.phase_shift)) if ctf.phase_shift is not None else 0.0,
        score=0.0,
        res_a=999.99,
        df_hand=df_hand,
    )


# ------------------------------------------------------------------ Warp (µm, degrees, π units)


def from_warp_values(defocus_um: float, defocus_delta_um: float, defocus_angle_deg: float, phase_pi: float):
    """Warp per-tilt CTF grid values -> ``CTFMetadata``: ``U/V = (Defocus ± Delta/2)·1e4`` Å, phase ``·180``."""
    return ctf_metadata(
        defocus_u_a=(defocus_um + defocus_delta_um / 2.0) * 1e4,
        defocus_v_a=(defocus_um - defocus_delta_um / 2.0) * 1e4,
        defocus_angle_deg=defocus_angle_deg,
        phase_shift_deg=phase_pi * 180.0,
    )


def to_warp_values(ctf) -> Tuple[float, float, float, float]:
    """``CTFMetadata`` -> ``(defocus_um, defocus_delta_um, defocus_angle_deg, phase_pi)``."""
    u, v = float(ctf.defocus_u), float(ctf.defocus_v)
    phase = 0.0 if ctf.phase_shift is None else float(ctf.phase_shift) / 180.0
    return (u + v) / 2.0 * 1e-4, (u - v) * 1e-4, float(ctf.defocus_angle), phase


# ------------------------------------------------------------------ portal PerSectionParameters (Å, radians)


def from_portal_values(
    major_defocus_a: float,
    minor_defocus_a: float,
    astigmatic_angle_deg: Optional[float],
    phase_shift_rad: Optional[float],
):
    return ctf_metadata(
        defocus_u_a=major_defocus_a,
        defocus_v_a=minor_defocus_a,
        defocus_angle_deg=astigmatic_angle_deg or 0.0,
        phase_shift_deg=None if phase_shift_rad is None else math.degrees(phase_shift_rad),
    )
