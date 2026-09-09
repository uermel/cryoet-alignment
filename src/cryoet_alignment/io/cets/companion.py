"""The companion manifest ``<doc>.cets-companion.json`` (schema ``cets-rigid-companion/0.2``).

Explicitly OUTSIDE CETS: everything a native project needs that the CETS document cannot carry —
acquisition order and per-image exposure (CETS stores only the pre-exposure, so the last exposure is
unrecoverable), nominal-vs-refined angles, kV / Cs / amplitude contrast, defocus hand, AlphaOffset /
BetaOffset, Warp's ``AreAnglesInverted`` and pixel sizes, FlipVol, alignment ↔ tomogram bindings, header vs
implied voxel sizes, annotation provenance (portal metadata, star flavour and pixel sizes, preserved star
columns), and what was dropped. Keyed by CETS ids. Writers read it when present (values are
reported with provenance ``companion``) and require the inputs otherwise; it never overrides geometry.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cryoet_alignment.io.cets.profile import COMPANION_VERSION, COMPANION_VERSIONS_READABLE


class ImageCompanion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acquisition_index_1b: Optional[int] = None
    exposure_dose: Optional[float] = None
    stage_angle_deg: Optional[float] = None
    frame_name: Optional[str] = None
    use_tilt: Optional[bool] = None


class TiltSeriesCompanion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_tool: Optional[str] = None
    source_version: Optional[str] = None
    voltage_kv: Optional[float] = None
    cs_mm: Optional[float] = None
    amplitude_contrast: Optional[float] = None
    dose_rate: Optional[float] = None
    pixel_size_acquisition_a: Optional[float] = None
    pixel_size_ctf_a: Optional[float] = None
    tilt_axis_nominal_deg: Optional[float] = None
    alpha_offset_deg: Optional[float] = None
    beta_offset_deg: Optional[float] = None
    are_angles_inverted: Optional[bool] = None
    defocus_hand: Optional[int] = None
    defocus_hand_convention: Optional[str] = None
    collection_metadata_path: Optional[str] = None  # the acquisition mdoc (path relative to the document, or a URL)
    images: Dict[str, ImageCompanion] = Field(default_factory=dict)  # keyed by TiltImage.id


class AlignmentCompanion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    tilt_series_id: str
    format: Optional[str] = None
    alignment_type: Optional[str] = None
    method_type: Optional[str] = None
    is_portal_standard: Optional[bool] = None
    reference_tomogram_id: Optional[str] = None  # the tomogram whose frame the alignment is expressed in
    tomogram_ids: List[str] = Field(default_factory=list)  # every reconstruction bound to this alignment
    native_volume_dimension_a: Optional[Dict[str, float]] = None
    frame_convention: Optional[Dict[str, str]] = None
    dropped: List[str] = Field(default_factory=list)
    thickness_px: Optional[int] = None  # AreTomo3 '# Thickness' (estimated sample thickness, not the box depth)
    source_ref: Optional[str] = None


class TomogramCompanion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voxel_header_a: Optional[float] = None
    voxel_implied_a: Optional[float] = None
    flip_vol: Optional[int] = None
    processing: Optional[str] = None
    reconstruction_method: Optional[str] = None
    reconstruction_software: Optional[str] = None
    affine_transformation_matrix: Optional[List[List[float]]] = None
    volume_offset: Optional[Dict[str, float]] = None
    source_ref: Optional[str] = None


class AnnotationCompanion(BaseModel):
    """What CETS cannot hold about an annotation: provenance, portal metadata, star bookkeeping, preserved columns."""

    model_config = ConfigDict(extra="forbid")

    kind: Optional[str] = None  # points | oriented_points | instance_points | mask
    tomogram_id: Optional[str] = None
    source_tool: Optional[str] = None
    source_ref: Optional[str] = None
    name: Optional[str] = None
    # portal provenance (portal -> CETS); ``metadata`` is the ingestion metadata block verbatim
    portal_annotation_id: Optional[int] = None
    object_name: Optional[str] = None
    object_id: Optional[str] = None
    object_state: Optional[str] = None
    annotation_method: Optional[str] = None
    method_type: Optional[str] = None
    ground_truth_status: Optional[bool] = None
    is_curator_recommended: Optional[bool] = None
    annotation_software: Optional[str] = None
    confidence_precision: Optional[float] = None
    confidence_recall: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    shape_type: Optional[str] = None
    file_format: Optional[str] = None
    file_path: Optional[str] = None
    mrc_path: Optional[str] = None  # the MRC twin of a zarr mask
    voxel_spacing_a: Optional[float] = None
    mask_label: Optional[int] = None
    is_visualization_default: Optional[bool] = None
    # star provenance (stars -> CETS)
    flavour: Optional[str] = None
    coords_angpix_a: Optional[float] = None
    angpix_shifts_a: Optional[float] = None
    series_name: Optional[str] = None  # the rlnMicrographName / rlnTomoName value as written
    columns: Dict[str, List[Any]] = Field(default_factory=dict)  # preserved star columns, one value per point
    instance_ids: Optional[List[int]] = None
    dropped: List[str] = Field(default_factory=list)


class Companion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = COMPANION_VERSION
    generator: Optional[str] = None
    tilt_series: Dict[str, TiltSeriesCompanion] = Field(default_factory=dict)  # keyed by TiltSeries.id
    alignments: List[AlignmentCompanion] = Field(default_factory=list)
    tomograms: Dict[str, TomogramCompanion] = Field(default_factory=dict)  # keyed by Tomogram.id
    annotations: Dict[str, AnnotationCompanion] = Field(default_factory=dict)  # keyed by Annotation.id

    def alignment(self, tilt_series_id: Optional[str], name: Optional[str]) -> Optional[AlignmentCompanion]:
        for a in self.alignments:
            if a.tilt_series_id == tilt_series_id and a.name == name:
                return a
        return None

    @staticmethod
    def path_for(document_path: Union[str, Path]) -> Path:
        p = Path(document_path)
        stem = p.name[: -len(".cets.json")] if p.name.endswith(".cets.json") else p.stem
        return p.with_name(f"{stem}.cets-companion.json")

    def dump(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Companion":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") not in COMPANION_VERSIONS_READABLE:
            raise ValueError(
                f"{path}: companion schema {data.get('schema_version')!r}, expected one of {COMPANION_VERSIONS_READABLE}",
            )
        data["schema_version"] = COMPANION_VERSION
        return cls.model_validate(data)

    @classmethod
    def load_for(cls, document_path: Union[str, Path]) -> Optional["Companion"]:
        p = cls.path_for(document_path)
        return cls.load(p) if p.exists() else None
