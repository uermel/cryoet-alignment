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
# tilt angles.
WARP_TILT_ANGLE_SIGN: int = -1


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

    @classmethod
    def from_imod(
        cls,
        imod_alignment: ImodAlignment,
        vol: Optional[str] = None,
        vol_size: Tuple[float, float, float] = None,
    ):
        xf = imod_alignment.xf
        tlt = imod_alignment.tlt
        xtlt = imod_alignment.xtilt
        tiltcom = imod_alignment.tiltcom

        affine_transform = np.eye(4, 4).tolist()
        alignment_type = "GLOBAL"
        format = "IMOD"
        is_canonical = True
        tilt_offset = 0
        volume_offset = {"x": 0, "y": 0, "z": 0}
        x_rotation_offset = 0
        per_section_alignment_parameters = []

        if vol is not None:
            header = get_mrc_header_local(vol)
            x = header.cella.x / header.mx * header.nx
            z = header.cella.y / header.my * header.ny
            y = header.cella.z / header.mz * header.nz
        elif vol_size is not None:
            x = vol_size[0]
            y = vol_size[1]
            z = vol_size[2]
        else:
            x, y, z = 0, 0, 0

        volume_dimension = {"x": x, "y": y, "z": z}

        it = (
            zip(xf.alignments, tlt.angles, [0.0] * len(tlt.angles))
            if xtlt is None
            else zip(xf.alignments, tlt.angles, xtlt.angles)
        )

        skip = []
        if tiltcom is not None:
            skip = tiltcom.EXCLUDELIST2 if tiltcom.EXCLUDELIST2 is not None else []
            skip = [s - 1 for s in skip]

        for z_index, (xf_alignment, tilt_angle, xtlt_angle) in enumerate(it):
            if z_index in skip:
                continue

            volume_x_rotation = xtlt_angle

            in_plane_rotation, offset = imod2are(xf_alignment.rot_matrix(), xf_alignment.shift())
            x_offset, y_offset = offset[0], offset[1]

            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=z_index,
                    tilt_angle=tilt_angle,
                    volume_x_rotation=volume_x_rotation,
                    in_plane_rotation=in_plane_rotation.tolist(),
                    x_offset=x_offset,
                    y_offset=y_offset,
                ),
            )

        return cls(
            affine_transformation_matrix=affine_transform,
            alignment_type=alignment_type,
            format=format,
            is_portal_standard=is_canonical,
            tilt_offset=tilt_offset,
            volume_offset=volume_offset,
            x_rotation_offset=x_rotation_offset,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_imod_basename(cls, basename: str):
        imod_alignment = ImodAlignment.read(base_name=basename)
        vol_path = f"{basename}_full_rec.mrc" if os.path.exists(f"{basename}_full_rec.mrc") else None
        return cls.from_imod(imod_alignment=imod_alignment, vol=vol_path)

    @classmethod
    def from_aretomo3(cls, aln: AreTomo3ALN, vol: str = None, vol_size: Tuple[float, float, float] = None):
        affine_transform = np.eye(4, 4).tolist()
        alignment_type = "GLOBAL"
        format = "ARETOMO3"
        is_canonical = True
        tilt_offset = aln.AlphaOffset
        volume_offset = {"x": 0, "y": 0, "z": 0}
        x_rotation_offset = aln.BetaOffset
        per_section_alignment_parameters = []

        full_size = aln.RawSize[2]
        orig_sections = list(range(full_size))
        for d in aln.DarkFrames:
            orig_sections.remove(d.section_idx)
        assert len(orig_sections) == len(
            aln.GlobalAlignments,
        ), "Number of sections does not match number of DarkFrames."

        if vol is not None:
            header = get_mrc_header_local(vol)
            x = header.cella.x / header.mx * header.nx
            y = header.cella.y / header.my * header.ny
            z = header.cella.z / header.mz * header.nz
            volume_dimension = {"x": x, "y": y, "z": z}
        elif vol_size is not None:
            x = vol_size[0]
            y = vol_size[1]
            z = vol_size[2]
            volume_dimension = {"x": x, "y": y, "z": z}
        else:
            x, y, z = 0, 0, 0
            volume_dimension = {"x": x, "y": y, "z": z}

        for idx, ali in zip(orig_sections, aln.GlobalAlignments):
            z_index = idx
            tilt_angle = ali.tilt
            volume_x_rotation = 0
            in_plane_rotation = ang2mat(ali.rot).tolist()
            x_offset = ali.tx
            y_offset = ali.ty

            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=z_index,
                    tilt_angle=tilt_angle,
                    volume_x_rotation=volume_x_rotation,
                    in_plane_rotation=in_plane_rotation,
                    x_offset=x_offset,
                    y_offset=y_offset,
                ),
            )

        return cls(
            affine_transformation_matrix=affine_transform,
            alignment_type=alignment_type,
            format=format,
            is_portal_standard=is_canonical,
            tilt_offset=tilt_offset,
            volume_offset=volume_offset,
            x_rotation_offset=x_rotation_offset,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_aretomo3_basename(cls, basename: str):
        aln = AreTomo3ALN.from_file(f"{basename}.aln")
        vol_path = f"{basename}_Vol.mrc"
        return cls.from_aretomo3(vol=vol_path, aln=aln)

    @classmethod
    def from_warp(
        cls,
        warp: WarpAlignment,
        vol: Optional[str] = None,
        vol_size: Tuple[float, float, float] = None,
    ):
        """Convert a WarpAlignment to the canonical Alignment format.

        Args:
            warp: ``WarpAlignment`` to convert.
            vol: Optional path to a reconstructed MRC volume — its header determines
                the ``volume_dimension``. Mutually exclusive with ``vol_size``.
            vol_size: ``(x, y, z)`` volume dimensions in pixels. If neither ``vol`` nor
                ``vol_size`` is provided, derives from ``warp.volume_dimensions_physical``
                divided by ``warp.pixel_size_a``.

        Note:
            Warp local fields (per-image 2D warp grids and 3D volume warp grids) are
            not represented in ``Alignment`` and are dropped here.
        """
        affine_transform = np.eye(4, 4).tolist()
        alignment_type = "GLOBAL"
        format = "WARP"
        is_canonical = True
        tilt_offset = 0
        volume_offset = {"x": 0, "y": 0, "z": 0}
        x_rotation_offset = 0

        if vol is not None:
            header = get_mrc_header_local(vol)
            x = header.cella.x / header.mx * header.nx
            y = header.cella.y / header.my * header.ny
            z = header.cella.z / header.mz * header.nz
        elif vol_size is not None:
            x, y, z = vol_size[0], vol_size[1], vol_size[2]
        elif warp.pixel_size_a > 0 and len(warp.volume_dimensions_physical) == 3:
            x = warp.volume_dimensions_physical[0] / warp.pixel_size_a
            y = warp.volume_dimensions_physical[1] / warp.pixel_size_a
            z = warp.volume_dimensions_physical[2] / warp.pixel_size_a
        else:
            x, y, z = 0, 0, 0

        volume_dimension = {"x": x, "y": y, "z": z}

        per_section_alignment_parameters = []
        for e in warp.entries:
            x_offset = e.tilt_axis_offset_x / warp.pixel_size_a if warp.pixel_size_a > 0 else 0.0
            y_offset = e.tilt_axis_offset_y / warp.pixel_size_a if warp.pixel_size_a > 0 else 0.0
            in_plane_rotation = ang2mat(WARP_TILT_AXIS_SIGN * e.tilt_axis_angle).tolist()

            per_section_alignment_parameters.append(
                PerSectionAlignmentParameters(
                    z_index=e.z_index,
                    tilt_angle=WARP_TILT_ANGLE_SIGN * e.tilt_angle,
                    volume_x_rotation=0.0,
                    in_plane_rotation=in_plane_rotation,
                    x_offset=x_offset,
                    y_offset=y_offset,
                ),
            )

        return cls(
            affine_transformation_matrix=affine_transform,
            alignment_type=alignment_type,
            format=format,
            is_portal_standard=is_canonical,
            tilt_offset=tilt_offset,
            volume_offset=volume_offset,
            x_rotation_offset=x_rotation_offset,
            per_section_alignment_parameters=per_section_alignment_parameters,
            volume_dimension=volume_dimension,
        )

    @classmethod
    def from_warp_file(
        cls,
        xml_path: PATH_TYPE,
        vol: Optional[str] = None,
        vol_size: Tuple[float, float, float] = None,
        pixel_size_a: Optional[float] = None,
    ):
        """Load a Warp XML and convert it to canonical Alignment format.

        ``pixel_size_a`` (Å/px) is passed through to ``WarpAlignment.from_file``
        and is required to recover per-tilt shift offsets in pixel units —
        the Warp XML stores them in Å and we need to divide by the image
        pixel size to round-trip them through AreTomo's ``tx``/``ty``.
        """
        warp = WarpAlignment.from_file(xml_path, pixel_size_a=pixel_size_a)
        return cls.from_warp(warp=warp, vol=vol, vol_size=vol_size)

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

        # To determine the excluded sections, we need to know the z_index of each section
        secs = {p.z_index: p for p in self.per_section_alignment_parameters}

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

        # Create the IMOD alignment files
        xf = ImodXF(alignments=xf_info)
        tlt = ImodTLT(angles=tlt_info)
        xtlt = ImodXTILT(angles=xtlt_info)

        # Output paths
        base = "basename" if basename is None else basename
        tilt_series_path = f"{base}.mrc"
        aligned_tilt_series_path = f"{base}_ali.mrc"
        volume_path = f"{base}_full_rec.mrc"
        xf_path = f"{base}.xf"
        tlt_path = f"{base}.tlt"
        xtlt_path = f"{base}.xtilt"

        if ts_spacing is not None:
            # Thickness in unbinned images
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
    ):
        # To determine the excluded sections, we need to know the z_index of each section
        secs = {p.z_index: p for p in self.per_section_alignment_parameters}
        dark_frames = []
        for z_index in range(ts_size[2]):
            if z_index not in secs:
                dark_frames.append(DarkFrameInfo(section_idx=z_index, val2=0, angle=0))

        raw_size = ts_size
        num_patches = 0
        alpha_offset = self.tilt_offset
        beta_offset = self.x_rotation_offset
        global_alignments = []

        for p in self.per_section_alignment_parameters:
            global_alignments.append(
                GlobalAlignmentInfo(
                    sec=p.z_index + 1,  # AreTomo3 uses 1-based indexing
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
            RawSize=raw_size,
            NumPatches=num_patches,
            DarkFrames=dark_frames,
            AlphaOffset=alpha_offset,
            BetaOffset=beta_offset,
            GlobalAlignments=global_alignments,
        )

    def to_warp(
        self,
        pixel_size_a: float,
        image_size_px: Tuple[int, int],
    ) -> WarpAlignment:
        """Convert the alignment to Warp/warpylib format.

        Args:
            pixel_size_a: Pixel size in Angstroms (Å/px). Used to convert pixel-space
                offsets to Angstrom-space ``tilt_axis_offset_{x,y}`` and to compute
                physical image/volume dimensions.
            image_size_px: ``(W, H)`` of the tilt images in pixels — used for
                ``image_dimensions_physical``.
        """
        entries: List[WarpAlignmentEntry] = []
        for p in self.per_section_alignment_parameters:
            entries.append(
                WarpAlignmentEntry(
                    z_index=p.z_index,
                    tilt_angle=WARP_TILT_ANGLE_SIGN * p.tilt_angle,
                    tilt_axis_angle=WARP_TILT_AXIS_SIGN * p.tilt_axis_rotation,
                    tilt_axis_offset_x=p.x_offset * pixel_size_a,
                    tilt_axis_offset_y=p.y_offset * pixel_size_a,
                ),
            )

        image_dims = [image_size_px[0] * pixel_size_a, image_size_px[1] * pixel_size_a]
        volume_dims = [
            self.volume_dimension["x"] * pixel_size_a,
            self.volume_dimension["y"] * pixel_size_a,
            self.volume_dimension["z"] * pixel_size_a,
        ]

        return WarpAlignment(
            n_tilts=len(entries),
            pixel_size_a=pixel_size_a,
            image_dimensions_physical=image_dims,
            volume_dimensions_physical=volume_dims,
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
