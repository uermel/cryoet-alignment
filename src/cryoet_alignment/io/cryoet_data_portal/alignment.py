"""The canonical (hub) rigid tilt-series alignment: the cryoET Data Portal ``alignment_metadata.json`` model.

Every other format converts through this model. Its per-section parameters are in the AreTomo3
convention (verified live against portal alignment 18924 / dataset 10445):

* ``tilt_angle`` — refined tilt in degrees (AreTomo3 ``TILT``; AlphaOffset included);
* ``in_plane_rotation`` — 2×2 matrix of the tilt-axis rotation ``ROT`` (``ang2mat``);
* ``x_offset`` / ``y_offset`` — image-space shifts in **pixels of the tilt series** (``TX``/``TY``),
  applied after the rotation (the portal schema's "angstrom" docstring is wrong);
* ``volume_x_rotation`` — rotation about the in-plane axis perpendicular to the tilt axis (RELION
  ``rlnTomoXTilt``, Warp ``LevelAngleX``); AreTomo3 cannot represent it;
* ``z_index`` — 0-based raw-stack section; sections without an entry are dark/excluded.

Geometry (operator form, standard right-handed active matrices, see ``io.relion.alignment``):
``q_px = Rz(ROT) · Ry(tilt) · Rx(xrot) · p_px_centered  + (x_offset, y_offset) + image_centre``
with the AreTomo3/Warp centre convention ``N/2`` (float) for image and volume. ``volume_dimension`` is
the native reconstruction box in **Ångström** on every path (portal semantics); ``volume_offset`` and
``affine_transformation_matrix`` are portal registration fields and are carried, never applied.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.aretomo3.aln import DarkFrameInfo, GlobalAlignmentInfo
from cryoet_alignment.io.base import PATH_TYPE, FileIOBase
from cryoet_alignment.io.imod import ImodAlignment, ImodNEWSTCOM, ImodTILTCOM, ImodTLT, ImodXF, ImodXTILT
from cryoet_alignment.io.imod.xf import ImodXFInfo
from cryoet_alignment.io.relion import RelionAlignment, RelionAlignmentEntry
from cryoet_alignment.io.relion.alignment import rot_x, rot_y, rot_z
from cryoet_alignment.io.warp import WarpAlignment, WarpAlignmentEntry
from cryoet_alignment.util.image import get_mrc_header_local

# Sign convention for the Warp tilt-axis-angle field, validated against AreTomo's
# .aln ROT column on a known-good reconstruction. Flip to -1 if a future validation
# experiment determines that the AreTomo ↔ Warp recipe needs a sign inversion. This
# is the single source of truth for that convention across the library.
WARP_TILT_AXIS_SIGN: int = 1

# Sign convention for the Warp tilt-angle field. Empirically the Warp reconstruction
# convention is opposite to AreTomo's for the tilt angle (independent of the in-plane
# rotation handled by WARP_TILT_AXIS_SIGN). Applied symmetrically in to_warp and
# from_warp so the AreTomo round-trip still sees the original AreTomo-coordinate
# tilt angles. Composite: hub tilt = -(Angle + LevelAngleY), xrot = LevelAngleX
# (arewarpion conventions.py:114-119, TiltSeries.cs:429-433).
WARP_TILT_ANGLE_SIGN: int = -1

#: Tolerance (degrees) within which per-section X rotations must agree to be written as
#: Warp's single ``LevelAngleX``.
X_ROTATION_TOL_DEG: float = 1e-3


def ang2mat(angle):
    angle = np.radians(angle)
    mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return mat


def mat2ang(mat):
    return np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))


def are2imod(mat, shift):
    mat = mat.transpose()
    shift = mat @ (-1 * np.array(shift))
    return mat, shift


def imod2are(mat, shift):
    mat = mat.transpose()
    shift = mat @ -shift
    return mat, shift


def _volume_dimension_a(
    vol: Optional[str],
    vol_size_px: Optional[Tuple[float, float, float]],
    pixel_size_a: Optional[float],
    fallback_a: Optional[Tuple[float, float, float]] = None,
    swap_yz: bool = False,
) -> Dict[str, float]:
    """The native volume box in Å from an MRC header, a pixel box + pixel size, or a fallback."""
    if vol is not None:
        header = get_mrc_header_local(vol)
        x = header.cella.x / header.mx * header.nx
        y = header.cella.y / header.my * header.ny
        z = header.cella.z / header.mz * header.nz
        if swap_yz:
            y, z = z, y
        return {"x": float(x), "y": float(y), "z": float(z)}
    if vol_size_px is not None:
        if pixel_size_a is None:
            raise ValueError("vol_size_px is in pixels: pass pixel_size_a (Å/px) to express the volume in Å")
        return {"x": vol_size_px[0] * pixel_size_a, "y": vol_size_px[1] * pixel_size_a, "z": vol_size_px[2] * pixel_size_a}
    if fallback_a is not None:
        return {"x": float(fallback_a[0]), "y": float(fallback_a[1]), "z": float(fallback_a[2])}
    return {"x": 0.0, "y": 0.0, "z": 0.0}


class PerSectionAlignmentParameters(BaseModel):
    z_index: int
    tilt_angle: float
    volume_x_rotation: float
    in_plane_rotation: List[List[float]]
    x_offset: float
    y_offset: float

    @property
    def tilt_axis_rotation(self) -> float:
        return mat2ang(np.array(self.in_plane_rotation))

    @tilt_axis_rotation.setter
    def tilt_axis_rotation(self, value: float):
        self.in_plane_rotation = ang2mat(value).tolist()

    def rotation_matrix(self) -> np.ndarray:
        """The 3×3 operator ``Rz(ROT) · Ry(tilt) · Rx(xrot)`` (active, right-handed)."""
        return rot_z(self.tilt_axis_rotation) @ rot_y(self.tilt_angle) @ rot_x(self.volume_x_rotation)


class Alignment(FileIOBase):
    affine_transformation_matrix: List[List[float]]
    alignment_type: str
    format: str
    is_portal_standard: bool
    tilt_offset: float
    volume_offset: Dict[str, float]
    x_rotation_offset: float
    per_section_alignment_parameters: List[PerSectionAlignmentParameters]
    volume_dimension: Dict[str, float]

    @classmethod
    def from_string(cls, text: str):
        return cls(**json.loads(text))

    def __str__(self):
        return json.dumps(self.model_dump())

    # ------------------------------------------------------------------ helpers

    @property
    def sections(self) -> Dict[int, PerSectionAlignmentParameters]:
        return {p.z_index: p for p in self.per_section_alignment_parameters}

    @property
    def has_x_rotation(self) -> bool:
        return any(abs(p.volume_x_rotation) > 0.0 for p in self.per_section_alignment_parameters)

    def common_x_rotation(self, tol_deg: float = X_ROTATION_TOL_DEG) -> float:
        """The single X rotation shared by every section, or raise when they differ by more than ``tol_deg``."""
        values = [p.volume_x_rotation for p in self.per_section_alignment_parameters]
        if not values:
            return 0.0
        lo, hi = min(values), max(values)
        if hi - lo > tol_deg:
            raise ValueError(
                f"per-section volume_x_rotation varies from {lo:.4f} to {hi:.4f} deg (> {tol_deg} deg); "
                "the target can only represent one X rotation for the whole series",
            )
        return float(np.mean(values))

    # ------------------------------------------------------------------ IMOD

    @classmethod
    def from_imod(
        cls,
        imod_alignment: ImodAlignment,
        vol: Optional[str] = None,
        vol_size_px: Optional[Tuple[float, float, float]] = None,
        pixel_size_a: Optional[float] = None,
    ):
        """Convert an IMOD alignment (``.xf`` + ``.tlt`` [+ ``.xtilt``, ``tilt.com``]).

        Args:
            imod_alignment: The IMOD files.
            vol: Optional ``*_full_rec.mrc`` — IMOD writes it XZY (``tilt`` output before ``trimvol``),
                so its header ``ny`` is the thickness; y/z are swapped when read.
            vol_size_px: ``(x, y, z)`` volume box in pixels; requires ``pixel_size_a``.
            pixel_size_a: Å/px of the tilt series (only used with ``vol_size_px``).
        """
        xf = imod_alignment.xf
        tlt = imod_alignment.tlt
        xtlt = imod_alignment.xtilt
        tiltcom = imod_alignment.tiltcom

        volume_dimension = _volume_dimension_a(vol, vol_size_px, pixel_size_a, swap_yz=True)

        it = (
            zip(xf.alignments, tlt.angles, [0.0] * len(tlt.angles))
            if xtlt is None
            else zip(xf.alignments, tlt.angles, xtlt.angles)
        )

        skip = []
        if tiltcom is not None:
            skip = tiltcom.EXCLUDELIST2 if tiltcom.EXCLUDELIST2 is not None else []
            skip = [s - 1 for s in skip]

        per_section_alignment_parameters = []
        for z_index, (xf_alignment, tilt_angle, xtlt_angle) in enumerate(it):
            if z_index in skip:
                continue
            in_plane_rotation, offset = imod2are(xf_alignment.rot_matrix(), xf_alignment.shift())
            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=z_index,
                    tilt_angle=tilt_angle,
                    volume_x_rotation=xtlt_angle,
                    in_plane_rotation=in_plane_rotation.tolist(),
                    x_offset=offset[0],
                    y_offset=offset[1],
                ),
            )

        return cls(
            affine_transformation_matrix=np.eye(4, 4).tolist(),
            alignment_type="GLOBAL",
            format="IMOD",
            is_portal_standard=True,
            tilt_offset=0,
            volume_offset={"x": 0, "y": 0, "z": 0},
            x_rotation_offset=0,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_imod_basename(cls, basename: str):
        imod_alignment = ImodAlignment.read(base_name=basename)
        vol_path = f"{basename}_full_rec.mrc" if os.path.exists(f"{basename}_full_rec.mrc") else None
        return cls.from_imod(imod_alignment=imod_alignment, vol=vol_path)

    # ------------------------------------------------------------------ AreTomo3

    @classmethod
    def from_aretomo3(
        cls,
        aln: AreTomo3ALN,
        vol: Optional[str] = None,
        vol_size_px: Optional[Tuple[float, float, float]] = None,
        pixel_size_a: Optional[float] = None,
    ):
        """Convert an AreTomo3 ``.aln`` (global rows only; patch locals are not representable here).

        ``z_index`` is the row's ``SEC - 1`` (the 1-based tilt-sorted raw index AreTomo3 preserves across
        dark-frame removal); ``tilt_angle = TILT``, ``in_plane_rotation = ang2mat(ROT)``, offsets ``TX``/``TY``
        in pixels; ``tilt_offset = AlphaOffset``, ``x_rotation_offset = BetaOffset`` (provenance only —
        BetaOffset never enters AreTomo3's projection geometry).
        """
        fallback = None
        if aln.Thickness and pixel_size_a is not None:
            fallback = (aln.RawSize[0] * pixel_size_a, aln.RawSize[1] * pixel_size_a, aln.Thickness * pixel_size_a)
        volume_dimension = _volume_dimension_a(vol, vol_size_px, pixel_size_a, fallback_a=fallback)

        per_section_alignment_parameters = []
        for z_index, ali in zip(aln.z_indices(), aln.GlobalAlignments):
            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=z_index,
                    tilt_angle=ali.tilt,
                    volume_x_rotation=0.0,
                    in_plane_rotation=ang2mat(ali.rot).tolist(),
                    x_offset=ali.tx,
                    y_offset=ali.ty,
                ),
            )

        return cls(
            affine_transformation_matrix=np.eye(4, 4).tolist(),
            alignment_type="GLOBAL" if aln.is_rigid else "LOCAL",
            format="ARETOMO3",
            is_portal_standard=True,
            tilt_offset=aln.AlphaOffset,
            volume_offset={"x": 0, "y": 0, "z": 0},
            x_rotation_offset=aln.BetaOffset,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_aretomo3_basename(cls, basename: str):
        aln = AreTomo3ALN.from_file(f"{basename}.aln")
        vol_path = f"{basename}_Vol.mrc"
        return cls.from_aretomo3(vol=vol_path, aln=aln)

    # ------------------------------------------------------------------ Warp

    @classmethod
    def from_warp(
        cls,
        warp: WarpAlignment,
        vol: Optional[str] = None,
        vol_size_px: Optional[Tuple[float, float, float]] = None,
        pixel_size_a: Optional[float] = None,
        allow_varying_grids: bool = False,
    ):
        """Convert a ``WarpAlignment`` (rigid subset) to the canonical model.

        Closed form (arewarpion ``conventions.py``): ``tilt = -(Angle + LevelAngleY)``,
        ``ROT = AxisAngle``, ``xrot = LevelAngleX``, offsets = the effective image-space shift
        (``AxisOffset - constant GridMovement``) plus the projection of a constant ``GridVolumeWarp``
        (``R_t · w``), divided by the tilt-image pixel size. ``UseTilt=False`` rows are omitted
        (dark). ``tilt_offset`` records ``-LevelAngleY`` for provenance (the tilt already includes it).

        Args:
            warp: The parsed XML.
            vol / vol_size_px / pixel_size_a: Override the volume box (Å); default is the XML's
                ``VolumeDimensionsAngstrom``.
            allow_varying_grids: Proceed (dropping the deformation) when the XML carries spatially
                varying movement / volume-warp grids. Refused by default.
        """
        if warp.grid_audit.has_varying_grids and not allow_varying_grids:
            raise ValueError(
                "the Warp XML carries spatially varying deformation grids "
                f"({', '.join(warp.grid_audit.varying_grid_names)}); the canonical rigid model cannot "
                "represent them — pass allow_varying_grids=True to drop them explicitly",
            )
        fallback = None
        if all(v > 0 for v in warp.volume_dimensions_physical):
            fallback = tuple(warp.volume_dimensions_physical)
        volume_dimension = _volume_dimension_a(vol, vol_size_px, pixel_size_a, fallback_a=fallback)
        if all(v == 0 for v in volume_dimension.values()):
            raise ValueError("volume box unknown: the XML has zero VolumeDimensionsAngstrom and no override was given")

        s = warp.pixel_size_a
        w = np.array(warp.grid_audit.constant_volume_warp, dtype=np.float64)
        per_section_alignment_parameters = []
        for e in warp.entries:
            if not e.use_tilt:
                continue
            tilt = WARP_TILT_ANGLE_SIGN * (e.tilt_angle + warp.level_angle_y)
            rot = WARP_TILT_AXIS_SIGN * e.tilt_axis_angle
            xrot = warp.level_angle_x
            r = rot_z(rot) @ rot_y(tilt) @ rot_x(xrot)
            rw = r @ w
            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=e.z_index,
                    tilt_angle=tilt,
                    volume_x_rotation=xrot,
                    in_plane_rotation=ang2mat(rot).tolist(),
                    x_offset=(e.effective_offset_x + rw[0]) / s,
                    y_offset=(e.effective_offset_y + rw[1]) / s,
                ),
            )

        return cls(
            affine_transformation_matrix=np.eye(4, 4).tolist(),
            alignment_type="GLOBAL" if warp.is_rigid else "LOCAL",
            format="WARP",
            is_portal_standard=True,
            tilt_offset=-warp.level_angle_y,
            volume_offset={"x": 0, "y": 0, "z": 0},
            x_rotation_offset=0,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_warp_file(
        cls,
        xml_path: PATH_TYPE,
        vol: Optional[str] = None,
        vol_size_px: Optional[Tuple[float, float, float]] = None,
        pixel_size_a: Optional[float] = None,
        image_dims_a: Optional[List[float]] = None,
        volume_dims_a: Optional[List[float]] = None,
        allow_varying_grids: bool = False,
    ):
        """Load a Warp XML and convert it (see ``WarpAlignment.from_file`` for the dimension overrides)."""
        warp = WarpAlignment.from_file(
            xml_path, pixel_size_a=pixel_size_a, image_dims_a=image_dims_a, volume_dims_a=volume_dims_a,
        )
        return cls.from_warp(
            warp=warp, vol=vol, vol_size_px=vol_size_px, pixel_size_a=pixel_size_a, allow_varying_grids=allow_varying_grids,
        )

    # ------------------------------------------------------------------ RELION

    @classmethod
    def from_relion(
        cls,
        relion: RelionAlignment,
        vol: Optional[str] = None,
        vol_size_px: Optional[Tuple[float, float, float]] = None,
    ):
        """Convert a RelionAlignment to the canonical Alignment format.

        Label-level mapping, identical to RELION's own AreTomo importer
        (align_tiltseries_runner.cpp): ``tilt_angle = rlnTomoYTilt``,
        ``tilt_axis_rotation = rlnTomoZRot``, ``volume_x_rotation = rlnTomoXTilt``,
        offsets = shift_angst / pixel_size (Angstrom -> px). No sign flips and no
        center corrections (for ODD tomogram/image dimensions the RELION and
        AreTomo/Warp projection centers differ by up to 0.5 px). The volume box
        is ``rlnTomoSizeX/Y/Z × pixel size`` in Å unless overridden.
        """
        fallback = tuple(v * relion.pixel_size_a for v in relion.volume_size_px)
        volume_dimension = _volume_dimension_a(vol, vol_size_px, relion.pixel_size_a, fallback_a=fallback)

        per_section_alignment_parameters = []
        for e in relion.entries:
            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=e.z_index,
                    tilt_angle=e.y_tilt,
                    volume_x_rotation=e.x_tilt,
                    in_plane_rotation=ang2mat(e.z_rot).tolist(),
                    x_offset=e.x_shift_angst / relion.pixel_size_a,
                    y_offset=e.y_shift_angst / relion.pixel_size_a,
                ),
            )

        return cls(
            affine_transformation_matrix=np.eye(4, 4).tolist(),
            alignment_type="GLOBAL",
            format="RELION",
            is_portal_standard=True,
            tilt_offset=0,
            volume_offset={"x": 0, "y": 0, "z": 0},
            x_rotation_offset=0,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_relion_file(
        cls,
        tomograms_star: PATH_TYPE,
        tomo_name: str = None,
        image_size_px: Tuple[int, int] = None,
        vol: Optional[str] = None,
        vol_size_px: Optional[Tuple[float, float, float]] = None,
    ):
        """Load a RELION tomograms.star and convert one tomogram to the canonical
        Alignment format. ``tomo_name`` selects the tomogram when the file lists
        several; ``image_size_px`` is required only for matrix-only tables (see
        ``RelionAlignment.from_file``)."""
        relion = RelionAlignment.from_file(tomograms_star, tomo_name=tomo_name, image_size_px=image_size_px)
        return cls.from_relion(relion, vol=vol, vol_size_px=vol_size_px)

    # ------------------------------------------------------------------ writers

    def to_imod(
        self,
        ts_size: Tuple[int, int, int],
        ts_spacing: float = None,
        binning: int = 1,
        basename: str = None,
    ):
        """Convert the alignment to IMOD format.

        Args:
            ts_size (Tuple[int, int, int]): The size of the tilt series in pixels/sections, (x, y, sections)
            ts_spacing (float): The spacing of the tilt series in Angstrom.
            binning (int): The binning factor.
            basename (str): The base name of the output files.
        """
        xf_info = []
        tlt_info = []
        xtlt_info = []
        exclude = []

        secs = self.sections
        for z_index in range(ts_size[2]):
            if z_index not in secs:
                tlt_info.append(0)
                xtlt_info.append(0)
                xf_info.append(ImodXFInfo(mxx=1, mxy=0, myx=0, myy=1, sx=0, sy=0))
                exclude.append(str(z_index + 1))
            else:
                tlt_info.append(secs[z_index].tilt_angle)
                xtlt_info.append(secs[z_index].volume_x_rotation)
                mat = np.array(secs[z_index].in_plane_rotation)
                mat, shift = are2imod(mat, [secs[z_index].x_offset, secs[z_index].y_offset])
                xf_info.append(
                    ImodXFInfo(mxx=mat[0, 0], mxy=mat[0, 1], myx=mat[1, 0], myy=mat[1, 1], sx=shift[0], sy=shift[1]),
                )

        xf = ImodXF(alignments=xf_info)
        tlt = ImodTLT(angles=tlt_info)
        xtlt = ImodXTILT(angles=xtlt_info)

        base = "basename" if basename is None else basename
        tilt_series_path = f"{base}.mrc"
        aligned_tilt_series_path = f"{base}_ali.mrc"
        volume_path = f"{base}_full_rec.mrc"
        xf_path = f"{base}.xf"
        tlt_path = f"{base}.tlt"
        xtlt_path = f"{base}.xtilt"

        if ts_spacing is not None:
            # Thickness in unbinned pixels (volume_dimension is Å)
            thickness = round(self.volume_dimension["z"] / ts_spacing)
            tiltcom = ImodTILTCOM(
                InputProjections=tilt_series_path,
                OutputFile=volume_path,
                TILTFILE=tlt_path,
                XTILTFILE=xtlt_path,
                THICKNESS=thickness,
                FULLIMAGE=(ts_size[0], ts_size[1]),
                EXCLUDELIST2=exclude,
            )
        else:
            tiltcom = None

        newstcom = ImodNEWSTCOM(
            AntialiasFilter=-1,
            InputFile=tilt_series_path,
            OutputFile=aligned_tilt_series_path,
            TransformFile=xf_path,
            TaperAtFill=(0, 0),
            AdjustOrigin=True,
            OffsetsInXandY=(0.0, 0.0),
            ImagesAreBinned=1.0,
            BinByFactor=binning,
        )

        return ImodAlignment(
            xf=xf,
            tlt=tlt,
            xtilt=xtlt,
            tiltcom=tiltcom,
            newstcom=newstcom,
        )

    def to_aretomo(
        self,
        ts_size: Tuple[int, int, int],
        dark_angles: Optional[Dict[int, float]] = None,
        thickness_px: Optional[int] = None,
    ) -> AreTomo3ALN:
        """Write the rigid alignment as an AreTomo3 ``.aln`` (``NumPatches = 0``).

        Sections of ``range(ts_size[2])`` without parameters become ``# DarkFrame`` lines carrying the raw
        index, the 1-based SEC and the angle from ``dark_angles`` (0.0 when unknown — pass the nominal stage
        angles to keep them). ``volume_x_rotation`` is NOT representable in a ``.aln`` and is refused here;
        callers that want to drop it must zero it explicitly.
        """
        if self.has_x_rotation:
            worst = max(abs(p.volume_x_rotation) for p in self.per_section_alignment_parameters)
            raise ValueError(
                f"AreTomo3 .aln cannot represent a volume X rotation (max |xrot| = {worst:.4f} deg); "
                "drop it explicitly before calling to_aretomo",
            )
        secs = self.sections
        dark_angles = dark_angles or {}
        dark_frames = []
        for z_index in range(ts_size[2]):
            if z_index not in secs:
                dark_frames.append(
                    DarkFrameInfo(section_idx=z_index, val2=z_index + 1, angle=float(dark_angles.get(z_index, 0.0))),
                )

        global_alignments = []
        for p in sorted(self.per_section_alignment_parameters, key=lambda q: q.z_index):
            global_alignments.append(
                GlobalAlignmentInfo(
                    sec=p.z_index + 1,  # 1-based tilt-sorted raw index
                    rot=p.tilt_axis_rotation,
                    gmag=1.0,
                    tx=p.x_offset,
                    ty=p.y_offset,
                    smean=1.0,
                    sfit=1.0,
                    scale=1.0,
                    base=0.0,
                    tilt=p.tilt_angle,
                ),
            )

        return AreTomo3ALN(
            RawSize=tuple(int(v) for v in ts_size),
            NumPatches=0,
            DarkFrames=dark_frames,
            AlphaOffset=self.tilt_offset,
            BetaOffset=self.x_rotation_offset,
            Thickness=thickness_px,
            GlobalAlignments=global_alignments,
            LocalAlignments=[],
        )

    def to_warp(
        self,
        pixel_size_a: float,
        image_size_px: Tuple[int, int],
        n_rows: Optional[int] = None,
        dark_angles: Optional[Dict[int, float]] = None,
        doses: Optional[Dict[int, float]] = None,
        movie_paths: Optional[Dict[int, str]] = None,
        x_rotation_tol_deg: float = X_ROTATION_TOL_DEG,
    ) -> WarpAlignment:
        """Convert the alignment to Warp's rigid representation.

        One XML row per raw section ``0..n_rows-1`` (default: up to the largest ``z_index``), rows without
        parameters written with ``UseTilt=False`` and the angle from ``dark_angles``. The per-section
        X rotations must agree within ``x_rotation_tol_deg`` — Warp has one ``LevelAngleX`` for the
        series — otherwise a ``ValueError`` names the spread. ``LevelAngleY`` is written as 0 (it is a gauge;
        the hub tilt already includes any offset).

        Args:
            pixel_size_a: Tilt-image pixel size (Å/px): converts the pixel offsets to ``AxisOffset`` Å and
                sets ``ImageDimensionsAngstrom``.
            image_size_px: ``(W, H)`` of the tilt images.
            n_rows: Number of XML rows (raw sections). Must cover every ``z_index``.
            dark_angles: Hub-convention tilt angles for rows without parameters.
            doses: Accumulated dose before each row (e/Å²).
            movie_paths: ``MoviePath`` entry per row.
        """
        level_angle_x = self.common_x_rotation(x_rotation_tol_deg)
        secs = self.sections
        max_z = max(secs) if secs else -1
        if n_rows is None:
            n_rows = max_z + 1
        if max_z >= n_rows:
            raise ValueError(f"n_rows = {n_rows} does not cover z_index {max_z}")
        dark_angles = dark_angles or {}
        doses = doses or {}
        movie_paths = movie_paths or {}

        entries: List[WarpAlignmentEntry] = []
        for z in range(n_rows):
            p = secs.get(z)
            if p is None:
                entries.append(
                    WarpAlignmentEntry(
                        z_index=z,
                        tilt_angle=WARP_TILT_ANGLE_SIGN * float(dark_angles.get(z, 0.0)),
                        tilt_axis_angle=0.0,
                        tilt_axis_offset_x=0.0,
                        tilt_axis_offset_y=0.0,
                        use_tilt=False,
                        dose=float(doses.get(z, 0.0)),
                        movie_path=movie_paths.get(z, ""),
                    ),
                )
                continue
            entries.append(
                WarpAlignmentEntry(
                    z_index=z,
                    tilt_angle=WARP_TILT_ANGLE_SIGN * p.tilt_angle,
                    tilt_axis_angle=WARP_TILT_AXIS_SIGN * p.tilt_axis_rotation,
                    tilt_axis_offset_x=p.x_offset * pixel_size_a,
                    tilt_axis_offset_y=p.y_offset * pixel_size_a,
                    use_tilt=True,
                    dose=float(doses.get(z, 0.0)),
                    movie_path=movie_paths.get(z, ""),
                ),
            )

        image_dims = [image_size_px[0] * pixel_size_a, image_size_px[1] * pixel_size_a]
        volume_dims = [self.volume_dimension["x"], self.volume_dimension["y"], self.volume_dimension["z"]]

        return WarpAlignment(
            n_tilts=len(entries),
            pixel_size_a=pixel_size_a,
            pixel_size_source="explicit",
            ctf_pixel_size_a=pixel_size_a,
            image_dimensions_physical=image_dims,
            volume_dimensions_physical=volume_dims,
            level_angle_x=level_angle_x,
            level_angle_y=0.0,
            entries=entries,
        )

    def to_relion(
        self,
        tomo_name: str,
        pixel_size_a: float,
        hand: float = -1.0,
        voltage: float = 300.0,
        spherical_aberration: float = 2.7,
        amplitude_contrast: float = 0.07,
        pre_exposures: Optional[List[float]] = None,
    ) -> RelionAlignment:
        """Convert the alignment to RELION 5 format (GLOBAL alignment only).

        Inverse of ``from_relion``: ``rlnTomoYTilt = tilt_angle``,
        ``rlnTomoZRot = tilt_axis_rotation``, ``rlnTomoXTilt = volume_x_rotation``,
        shifts = offsets * pixel_size (px -> Angstrom); the refined tilt angle is
        also written as ``rlnTomoNominalStageTiltAngle`` (the canonical model
        carries no separate nominal angle). Absent ``z_index`` rows (dark tilts)
        are simply not emitted — RELION tables carry kept tilts only. The volume
        box (Å) becomes ``rlnTomoSizeX/Y/Z`` in pixels of ``pixel_size_a``.

        Args:
            tomo_name: rlnTomoName for the emitted tables.
            pixel_size_a: Tilt-series pixel size in Angstrom/px.
            hand: rlnTomoHand. Metadata passthrough — defaults to RELION's own
                import default (-1); override to match your data.
            voltage: rlnVoltage in kV (metadata passthrough, common Krios value).
            spherical_aberration: rlnSphericalAberration in mm (passthrough).
            amplitude_contrast: rlnAmplitudeContrast (passthrough).
            pre_exposures: Optional per-tilt cumulative dose in e/A^2, in the
                order of the emitted sections; zeros when omitted (RELION requires
                the column to exist).
        """
        params = sorted(self.per_section_alignment_parameters, key=lambda p: p.z_index)
        if pre_exposures is not None and len(pre_exposures) != len(params):
            raise ValueError(f"pre_exposures has {len(pre_exposures)} values for {len(params)} sections")

        entries: List[RelionAlignmentEntry] = []
        for i, p in enumerate(params):
            entries.append(
                RelionAlignmentEntry(
                    z_index=i,
                    nominal_stage_tilt_angle=p.tilt_angle,
                    x_tilt=p.volume_x_rotation,
                    y_tilt=p.tilt_angle,
                    z_rot=p.tilt_axis_rotation,
                    x_shift_angst=p.x_offset * pixel_size_a,
                    y_shift_angst=p.y_offset * pixel_size_a,
                    pre_exposure=pre_exposures[i] if pre_exposures is not None else 0.0,
                ),
            )

        return RelionAlignment(
            tomo_name=tomo_name,
            pixel_size_a=pixel_size_a,
            volume_size_px=(
                round(self.volume_dimension["x"] / pixel_size_a),
                round(self.volume_dimension["y"] / pixel_size_a),
                round(self.volume_dimension["z"] / pixel_size_a),
            ),
            hand=hand,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
            entries=entries,
        )

    def get_skipped_sections(self, ts_size: Tuple[int, int, int]):
        full_size = list(range(ts_size[2]))
        present_idx = [p.z_index for p in self.per_section_alignment_parameters]
        excl = []

        for idx in full_size:
            if idx not in present_idx:
                excl.append(idx)

        return excl

    def get_median_tilt_axis(self):
        return np.median([p.tilt_axis_rotation for p in self.per_section_alignment_parameters])
