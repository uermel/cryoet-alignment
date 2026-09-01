"""RELION 5 tomography tilt-series alignment format (GLOBAL alignment only).

RELION 5 stores tilt-series metadata in a ``tomograms.star`` with a
``data_global`` table (one row per tomogram) and one per-tomogram table with
per-tilt data. Two layouts exist and both are read here
(``src/jaz/tomography/tomogram_set.cpp:40-89``):

* **RELION-5 layout**: ``data_global`` carries ``rlnTomoTiltSeriesStarFile``
  pointing at a separate per-tomogram star file (resolved relative to the
  ``tomograms.star`` location when not absolute).
* **relion-4 / WarpTools layout**: the per-tomogram tables are embedded in the
  same file, named ``data_<rlnTomoName>``.

Per-tilt alignment is either Euler columns — ``rlnTomoXTilt`` (optional,
defaults to 0), ``rlnTomoYTilt``, ``rlnTomoZRot`` (degrees),
``rlnTomoXShiftAngst``/``rlnTomoYShiftAngst`` (Angstrom) — or the relion-4
projection-matrix rows ``rlnTomoProjX/Y/Z/W``. When all four matrix labels are
present RELION prefers the matrices (``tomogram_set.cpp:358-393``); this reader
then recovers Eulers and shifts exactly like RELION's own
``Tomogram::getProjectionAnglesFromMatrix`` (``tomogram.cpp:66-117``), which
requires the tilt-image dimensions (``image_size_px``) — a value no star file
carries. Writing always emits the RELION-5 layout with Euler columns, matching
what RELION 5 itself writes (it deactivates the matrix labels,
``tomogram_set.cpp:121-124``).

Conventions (label-level mapping, matching RELION's own AreTomo importer,
``align_tiltseries_runner.cpp:784-787,849-855``): angles in degrees, shifts in
Angstrom, one row per KEPT tilt (dark/excluded tilts are absent and cannot be
recovered), rows conventionally sorted by nominal stage tilt angle. No
geometric center corrections are applied: RELION's projection uses
integer-division volume/image centers while AreTomo/Warp center on N/2, so for
ODD tomogram or image dimensions the two geometries differ by up to 0.5 px —
use arewarpion for geometrically exact conversion in that case. Local
deformations, particle trajectories, and CTF columns are out of scope (CTF
columns are ignored on read and never written).

Note on relion-4/Warp matrix files: their matrices are often rotation-only
(zero translation, a centered-coordinate convention). The decomposition
reproduces RELION 5's own interpretation of such files — the recovered shifts
then absorb the centering offset, and re-composing the projection matrix from
the recovered values reproduces the original matrix in every
projection-relevant entry (the xy rows and the rotation; the z translation is
not representable in Euler+shift form and never enters a projection).
"""

import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import starfile
from pydantic import BaseModel

from cryoet_alignment.io.base import PATH_TYPE

VERSION_COMMENT = "# version 50001"

_EULER_LABELS = ("rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst")
_MATRIX_LABELS = ("rlnTomoProjX", "rlnTomoProjY", "rlnTomoProjZ", "rlnTomoProjW")


class RelionAlignmentEntry(BaseModel):
    """Per-tilt global alignment in RELION convention.

    Attributes:
        z_index: 0-based row index among the KEPT tilts (RELION tables carry
            one row per kept tilt, conventionally sorted by stage tilt angle).
        nominal_stage_tilt_angle: rlnTomoNominalStageTiltAngle in degrees
            (falls back to the refined y_tilt when the column is absent).
        x_tilt: rlnTomoXTilt in degrees (0 when absent).
        y_tilt: rlnTomoYTilt in degrees (the refined tilt angle).
        z_rot: rlnTomoZRot in degrees (the tilt-axis rotation).
        x_shift_angst: rlnTomoXShiftAngst in Angstrom.
        y_shift_angst: rlnTomoYShiftAngst in Angstrom.
        pre_exposure: rlnMicrographPreExposure in e/A^2 (0 when absent).
    """

    z_index: int
    nominal_stage_tilt_angle: float
    x_tilt: float = 0.0
    y_tilt: float
    z_rot: float
    x_shift_angst: float
    y_shift_angst: float
    pre_exposure: float = 0.0


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def projection_matrix(
    x_tilt: float,
    y_tilt: float,
    z_rot: float,
    x_shift_angst: float,
    y_shift_angst: float,
    volume_size_px: Tuple[int, int, int],
    image_size_px: Tuple[int, int],
    pixel_size_a: float,
) -> np.ndarray:
    """RELION 5 projection matrix — literal transcription of
    ``Tomogram::setProjectionMatrix`` (tomogram.cpp:30-64).

    ``P = s1 * s2 * Rz(zrot) * Ry(ytilt) * Rx(xtilt) * s0`` operating on
    corner-origin PIXEL coordinates, with INTEGER-DIVISION centers:
    ``s0 = T(-(w0//2, h0//2, d0//2))``, ``s2 = T((nx//2, ny//2, 0))`` and
    ``s1 = T((xshift_angst/pix, yshift_angst/pix, 0))``.
    """
    rotation = _rot_z(z_rot) @ _rot_y(y_tilt) @ _rot_x(x_tilt)
    c_int = np.array([d // 2 for d in volume_size_px], dtype=np.float64)
    i_int = np.array([image_size_px[0] // 2, image_size_px[1] // 2, 0.0], dtype=np.float64)
    shift_px = np.array([x_shift_angst / pixel_size_a, y_shift_angst / pixel_size_a, 0.0])
    p = np.eye(4, dtype=np.float64)
    p[:3, :3] = rotation
    p[:3, 3] = shift_px + i_int - rotation @ c_int
    return p


def eulers_from_matrix(
    p: np.ndarray,
    volume_size_px: Tuple[int, int, int],
    image_size_px: Tuple[int, int],
    pixel_size_a: float,
) -> Tuple[float, float, float, float, float]:
    """Recover (x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst) from a
    projection matrix — literal port of
    ``Tomogram::getProjectionAnglesFromMatrix`` (tomogram.cpp:66-117),
    including its gimbal-lock branches and its center asymmetry: the volume
    center is the FLOAT ``N/2`` (``tomogram_set.cpp:315``) while the image
    center is the INTEGER-DIVISION ``N//2`` (tomogram.cpp:103).
    """
    a = np.asarray(p, dtype=np.float64)
    if a[2, 0] < 1.0:
        if a[2, 0] > -1.0:
            theta_x = math.atan2(a[2, 1], a[2, 2])
            theta_y = math.asin(-a[2, 0])
            theta_z = math.atan2(a[1, 0], a[0, 0])
        else:  # A(2,0) = -1
            theta_x = 0.0
            theta_y = math.pi / 2.0
            theta_z = -math.atan2(-a[1, 2], a[1, 1])
    else:  # A(2,0) = +1
        theta_x = 0.0
        theta_y = -math.pi / 2.0
        theta_z = math.atan2(-a[1, 2], a[1, 1])

    x_tilt = math.degrees(theta_x)
    y_tilt = math.degrees(theta_y)
    z_rot = math.degrees(theta_z)

    c_float = np.array([d / 2.0 for d in volume_size_px], dtype=np.float64)
    i_int = np.array([image_size_px[0] // 2, image_size_px[1] // 2, 0.0], dtype=np.float64)

    s0 = np.eye(4)
    s0[:3, 3] = c_float
    s2 = np.eye(4)
    s2[:3, 3] = -i_int
    r_inv = np.eye(4)
    r_inv[:3, :3] = _rot_x(-x_tilt) @ _rot_y(-y_tilt) @ _rot_z(-z_rot)

    s1 = a @ s0 @ r_inv @ s2
    return x_tilt, y_tilt, z_rot, pixel_size_a * s1[0, 3], pixel_size_a * s1[1, 3]


def _parse_vector(value) -> List[float]:
    """``"[a,b,c,d]"`` bracket strings (as starfile hands them over) or iterables."""
    if isinstance(value, str):
        return [float(v) for v in value.strip().strip("[]").split(",")]
    return [float(v) for v in value]


def _validate_matrix(p: np.ndarray, row: int) -> None:
    if not np.isfinite(p).all():
        raise ValueError(f"non-finite projection matrix in row {row}")
    if not np.allclose(p[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"projection matrix row {row} is not homogeneous (last row != [0,0,0,1])")
    r = p[:3, :3]
    if np.abs(r @ r.T - np.eye(3)).max() > 1e-5:
        raise ValueError(f"projection matrix row {row} has a non-orthonormal linear part")
    if abs(np.linalg.det(r) - 1.0) > 1e-5:
        raise ValueError(f"projection matrix row {row} is not a proper rotation (det != +1)")


def _as_frame(block) -> pd.DataFrame:
    return block.to_frame().T if isinstance(block, pd.Series) else block


class RelionAlignment(BaseModel):
    """One tomogram's RELION 5 global tilt-series alignment.

    Attributes:
        tomo_name: rlnTomoName.
        pixel_size_a: rlnTomoTiltSeriesPixelSize in Angstrom/px.
        volume_size_px: (rlnTomoSizeX, rlnTomoSizeY, rlnTomoSizeZ) in unbinned px.
        hand: rlnTomoHand (+-1); None when absent.
        voltage: rlnVoltage in kV; None when absent.
        spherical_aberration: rlnSphericalAberration in mm; None when absent.
        amplitude_contrast: rlnAmplitudeContrast; None when absent.
        optics_group_name: rlnOpticsGroupName.
        entries: Per-tilt alignment entries in table order.
    """

    tomo_name: str
    pixel_size_a: float
    volume_size_px: Tuple[int, int, int]
    hand: Optional[float] = None
    voltage: Optional[float] = None
    spherical_aberration: Optional[float] = None
    amplitude_contrast: Optional[float] = None
    optics_group_name: str = "opticsGroup1"
    entries: List[RelionAlignmentEntry]

    @property
    def n_tilts(self) -> int:
        return len(self.entries)

    @classmethod
    def from_file(
        cls,
        tomograms_star: PATH_TYPE,
        tomo_name: str = None,
        image_size_px: Tuple[int, int] = None,
    ) -> "RelionAlignment":
        """Load one tomogram's alignment from a RELION tomograms.star.

        Args:
            tomograms_star: Path to the tomograms.star (either layout).
            tomo_name: Which tomogram to load when the file lists several
                (optional when it lists exactly one).
            image_size_px: Tilt-image dimensions in px — required ONLY when
                the per-tilt table carries projection matrices instead of
                Euler columns (the matrix decomposition needs the image
                center, which no star file stores).
        """
        path = Path(tomograms_star)
        blocks = starfile.read(path, always_dict=True)
        if "global" not in blocks:
            raise ValueError(f"{path} has no data_global block — not a tomograms.star")
        glob = _as_frame(blocks["global"])

        names = [str(n) for n in glob["rlnTomoName"]]
        if tomo_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"{path} lists {len(names)} tomograms — pass tomo_name (one of: {', '.join(names)})",
                )
            tomo_name = names[0]
        if tomo_name not in names:
            raise ValueError(f"tomogram {tomo_name!r} not in {path} (has: {', '.join(names)})")
        row = glob.iloc[names.index(tomo_name)]

        if "rlnTomoTiltSeriesStarFile" in glob.columns:
            ts_file = Path(str(row["rlnTomoTiltSeriesStarFile"]))
            if not ts_file.is_absolute():
                candidate = path.parent / ts_file
                if candidate.exists():
                    ts_file = candidate
            ts_blocks = starfile.read(ts_file, always_dict=True)
            tilt = _as_frame(ts_blocks[tomo_name] if tomo_name in ts_blocks else next(iter(ts_blocks.values())))
        elif tomo_name in blocks:
            tilt = _as_frame(blocks[tomo_name])
        else:
            raise ValueError(
                f"{path} has neither rlnTomoTiltSeriesStarFile nor an embedded data_{tomo_name} block",
            )

        volume_size_px = (
            int(row["rlnTomoSizeX"]),
            int(row["rlnTomoSizeY"]),
            int(row["rlnTomoSizeZ"]),
        )
        pixel_size_a = float(row["rlnTomoTiltSeriesPixelSize"])

        def opt(label):
            return float(row[label]) if label in glob.columns else None

        entries = cls._parse_tilt_table(tilt, volume_size_px, image_size_px, pixel_size_a)
        return cls(
            tomo_name=tomo_name,
            pixel_size_a=pixel_size_a,
            volume_size_px=volume_size_px,
            hand=opt("rlnTomoHand"),
            voltage=opt("rlnVoltage"),
            spherical_aberration=opt("rlnSphericalAberration"),
            amplitude_contrast=opt("rlnAmplitudeContrast"),
            optics_group_name=(
                str(row["rlnOpticsGroupName"]) if "rlnOpticsGroupName" in glob.columns else "opticsGroup1"
            ),
            entries=entries,
        )

    @staticmethod
    def _parse_tilt_table(
        tilt: pd.DataFrame,
        volume_size_px: Tuple[int, int, int],
        image_size_px: Optional[Tuple[int, int]],
        pixel_size_a: float,
    ) -> List[RelionAlignmentEntry]:
        has_matrices = all(c in tilt.columns for c in _MATRIX_LABELS)
        has_eulers = all(c in tilt.columns for c in _EULER_LABELS)
        use_matrices = has_matrices
        if has_matrices and image_size_px is None:
            if has_eulers:
                warnings.warn(
                    "rlnTomoProjX/Y/Z/W present but no image_size_px given — falling back to the "
                    "Euler columns (RELION itself would prefer the matrices)",
                    stacklevel=3,
                )
                use_matrices = False
            else:
                raise ValueError(
                    "per-tilt table carries only projection matrices (rlnTomoProjX/Y/Z/W); "
                    "pass image_size_px=(nx, ny) so the shifts can be recovered "
                    "(the matrix decomposition needs the tilt-image center)",
                )
        if not has_matrices and not has_eulers:
            missing = [c for c in _EULER_LABELS if c not in tilt.columns]
            raise ValueError(f"per-tilt table is missing mandatory columns: {', '.join(missing)}")

        def col(label, i, default=None):
            if label in tilt.columns:
                return float(tilt[label].iloc[i])
            return default

        entries = []
        for i in range(len(tilt)):
            if use_matrices:
                p = np.array([_parse_vector(tilt[c].iloc[i]) for c in _MATRIX_LABELS], dtype=np.float64)
                _validate_matrix(p, i)
                x_tilt, y_tilt, z_rot, sx, sy = eulers_from_matrix(
                    p,
                    volume_size_px,
                    image_size_px,
                    pixel_size_a,
                )
            else:
                x_tilt = col("rlnTomoXTilt", i, 0.0)
                y_tilt = col("rlnTomoYTilt", i)
                z_rot = col("rlnTomoZRot", i)
                sx = col("rlnTomoXShiftAngst", i)
                sy = col("rlnTomoYShiftAngst", i)
            entries.append(
                RelionAlignmentEntry(
                    z_index=i,
                    nominal_stage_tilt_angle=col("rlnTomoNominalStageTiltAngle", i, y_tilt),
                    x_tilt=x_tilt,
                    y_tilt=y_tilt,
                    z_rot=z_rot,
                    x_shift_angst=sx,
                    y_shift_angst=sy,
                    pre_exposure=col("rlnMicrographPreExposure", i, 0.0),
                ),
            )
        return entries

    def to_file(self, tomograms_star: PATH_TYPE) -> None:
        """Write the RELION-5 two-file layout: ``tomograms.star`` at the given
        path plus ``tilt_series/<tomo_name>.star`` next to it, referenced with
        a RELATIVE ``rlnTomoTiltSeriesStarFile`` path (portable trees).

        Requires ``hand``, ``voltage``, ``spherical_aberration`` and
        ``amplitude_contrast`` to be set — RELION refuses tomograms.star files
        without them, and this writer refuses to invent values.
        """
        missing = [
            n for n in ("hand", "voltage", "spherical_aberration", "amplitude_contrast") if getattr(self, n) is None
        ]
        if missing:
            raise ValueError(
                f"cannot write tomograms.star without {', '.join(missing)} — RELION requires them; "
                "set the fields (or pass them to Alignment.to_relion)",
            )

        path = Path(tomograms_star)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts_rel = Path("tilt_series") / f"{self.tomo_name}.star"
        ts_path = path.parent / ts_rel
        ts_path.parent.mkdir(parents=True, exist_ok=True)

        glob = pd.DataFrame(
            [
                {
                    "rlnTomoName": self.tomo_name,
                    "rlnVoltage": self.voltage,
                    "rlnSphericalAberration": self.spherical_aberration,
                    "rlnAmplitudeContrast": self.amplitude_contrast,
                    "rlnTomoHand": self.hand,
                    "rlnOpticsGroupName": self.optics_group_name,
                    "rlnTomoTiltSeriesPixelSize": self.pixel_size_a,
                    "rlnTomoSizeX": self.volume_size_px[0],
                    "rlnTomoSizeY": self.volume_size_px[1],
                    "rlnTomoSizeZ": self.volume_size_px[2],
                    "rlnTomoTiltSeriesStarFile": str(ts_rel),
                },
            ],
        )
        starfile.write({"global": glob}, path)

        tilt = pd.DataFrame(
            [
                {
                    "rlnTomoNominalStageTiltAngle": e.nominal_stage_tilt_angle,
                    "rlnTomoXTilt": e.x_tilt,
                    "rlnTomoYTilt": e.y_tilt,
                    "rlnTomoZRot": e.z_rot,
                    "rlnTomoXShiftAngst": e.x_shift_angst,
                    "rlnTomoYShiftAngst": e.y_shift_angst,
                    "rlnMicrographPreExposure": e.pre_exposure,
                }
                for e in self.entries
            ],
        )
        starfile.write({self.tomo_name: tilt}, ts_path)

        for f in (path, ts_path):
            f.write_text(f"{VERSION_COMMENT}\n" + f.read_text())
