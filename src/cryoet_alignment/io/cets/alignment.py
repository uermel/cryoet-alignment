"""The codec: hub ``Alignment`` (AreTomo3-convention per-section parameters) <-> CETS ``Alignment``.

Profile encoding of one projection (``docs/cets.md``)::

    ProjectionAlignment(id=<ts>_<aln>_align_<z>, tilt_image_id=<ts>_<z>, name="tomogram_to_projection",
                        input="physical", output="physical",
                        sequence=[Affine("tilt", Ry(tilt)·Rx(xrot)),
                                  Affine("in_plane_rotation", Rz(rot)),
                                  Translation("shift", [tx, ty])])          # Å

    q_img = drop_z(Rz · Ry · Rx · p_tomo) + shift

with ``p_tomo`` in the REFERENCE tomogram's centred physical frame (origin at array index floor(N/2), z up =
Warp/RELION z) and ``q_img`` in the tilt image's centred physical frame. The hub's shifts refer to the native
tool's frames (``FrameConvention``); with ``delta = native_centre - cets_centre`` (Å):

    shift_cets = shift_native + delta_image - (R · delta_volume)[:2]

and the inverse on the way back. Both directions take the frame, the image size/spacing and the reference
volume explicitly — nothing is left to callers' assumptions. The native volume box (hub ``volume_dimension``,
Å) must agree with the reference tomogram (``N_tomo · s_tomo``) within one voxel per axis; otherwise the
association is refused.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cryoet_alignment.io.cets.frames import (
    CETS_FRAME,
    FrameConvention,
    ImageFrame,
    centre_index,
    find_by_id,
    frame_convention_for,
    image_frame,
)
from cryoet_alignment.io.cets.profile import (
    IN_PLANE_ROTATION,
    PHYSICAL_CS,
    SHIFT,
    TILT,
    TOMOGRAM_TO_PROJECTION,
    projection_alignment_id,
    require_cets,
    tilt_image_id,
)
from cryoet_alignment.io.cets.rotation import check_rotation, decompose, in_plane_matrix, rotation, tilt_matrix
from cryoet_alignment.io.cryoet_data_portal.alignment import Alignment, PerSectionAlignmentParameters, ang2mat

#: Maximum disagreement (in reference-tomogram voxels, per axis) between the hub's native volume box and the
#: reference tomogram's physical extent.
VOLUME_TOLERANCE_VOXELS = 1.0


@dataclass(frozen=True)
class ReferenceVolume:
    """The tomogram entity that defines the CETS tomogram frame for an alignment."""

    size_px: Tuple[int, int, int]
    spacing_a: float

    @property
    def extent_a(self) -> np.ndarray:
        return np.array(self.size_px, dtype=np.float64) * self.spacing_a

    def cets_centre_a(self) -> np.ndarray:
        return np.array([centre_index(n, "floor") for n in self.size_px], dtype=np.float64) * self.spacing_a

    @classmethod
    def from_tomogram(cls, tomogram) -> "ReferenceVolume":
        fr = image_frame(tomogram)
        if fr.ndim != 3:
            raise ValueError("reference tomogram must be a 3D image")
        return cls(size_px=tuple(fr.size_px), spacing_a=fr.isotropic_spacing)


def reconcile_volume(native_dimension_a: Dict[str, float], reference: ReferenceVolume) -> None:
    """Refuse when the alignment's native box and the reference tomogram disagree by more than one voxel."""
    native = np.array([native_dimension_a["x"], native_dimension_a["y"], native_dimension_a["z"]], dtype=np.float64)
    diff = np.abs(native - reference.extent_a) / reference.spacing_a
    if np.any(diff > VOLUME_TOLERANCE_VOXELS):
        raise ValueError(
            "alignment volume box and reference tomogram disagree: native "
            f"{native.round(2).tolist()} Å vs tomogram {reference.extent_a.round(2).tolist()} Å "
            f"({reference.size_px} × {reference.spacing_a:.5f} Å); max deviation {diff.max():.2f} voxels "
            f"(tolerance {VOLUME_TOLERANCE_VOXELS}). Select the matching tomogram or pass the native box explicitly.",
        )


def _native_volume_centre_a(
    native_dimension_a: Dict[str, float],
    frame: FrameConvention,
    pixel_size_a: float,
) -> np.ndarray:
    v = np.array([native_dimension_a["x"], native_dimension_a["y"], native_dimension_a["z"]], dtype=np.float64)
    if frame.volume_center == "half":
        return v / 2.0
    # floor: the tool's volume box in its own pixels, integer-division centre (RELION, tomogram.cpp:44)
    n_px = np.round(v / pixel_size_a).astype(int)
    return np.floor(n_px / 2.0) * pixel_size_a


def _deltas(
    frame: FrameConvention,
    image: ImageFrame,
    reference: ReferenceVolume,
    native_dimension_a: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """``(delta_image [2], delta_volume [3])`` in Å: native centre minus CETS centre."""
    d_img = image.centre_delta_a(frame.image_center)[:2]
    d_vol = _native_volume_centre_a(native_dimension_a, frame, image.isotropic_spacing) - reference.cets_centre_a()
    return d_img, d_vol


# --------------------------------------------------------------------------- hub -> CETS


def alignment_to_cets(
    hub: Alignment,
    *,
    tilt_series_id: str,
    alignment_name: str,
    image: ImageFrame,
    reference: ReferenceVolume,
    frame: Optional[FrameConvention] = None,
    native_dimension_a: Optional[Dict[str, float]] = None,
    tilt_image_ids: Optional[Dict[int, str]] = None,
):
    """Encode a hub alignment as a profile CETS ``Alignment``.

    Args:
        hub: The canonical alignment (offsets in tilt-image pixels).
        tilt_series_id: ``TiltSeries.id`` the alignment belongs to.
        alignment_name: Alignment-instance name used in the projection-alignment ids.
        image: The tilt images' frame (size, spacing, origin) — from ``image_frame(tilt_image)`` or built by
            the caller; the spacing converts pixel offsets to Å.
        reference: The reference tomogram (size, voxel) that defines the tomogram frame.
        frame: Native centre convention of the hub's source; default ``frame_convention_for(hub.format)``.
        native_dimension_a: Override of the hub's native volume box (Å).
        tilt_image_ids: ``z_index -> TiltImage.id`` when the ids do not follow ``<ts>_<z>``.
    """
    m = require_cets()
    frame = frame or frame_convention_for(hub.format)
    native = native_dimension_a or hub.volume_dimension
    reconcile_volume(native, reference)
    s = image.isotropic_spacing
    d_img, d_vol = _deltas(frame, image, reference, native)

    projections = []
    for p in sorted(hub.per_section_alignment_parameters, key=lambda q: q.z_index):
        rot, tilt, xrot = p.tilt_axis_rotation, p.tilt_angle, p.volume_x_rotation
        r = rotation(rot, tilt, xrot)
        t_native = np.array([p.x_offset, p.y_offset], dtype=np.float64) * s
        t_cets = t_native + d_img - (r @ d_vol)[:2]
        ti_id = (tilt_image_ids or {}).get(p.z_index, tilt_image_id(tilt_series_id, p.z_index))
        projections.append(
            m.ProjectionAlignment(
                id=projection_alignment_id(tilt_series_id, alignment_name, p.z_index),
                tilt_image_id=ti_id,
                name=TOMOGRAM_TO_PROJECTION,
                input=PHYSICAL_CS,
                output=PHYSICAL_CS,
                sequence=[
                    m.Affine(name=TILT, affine=tilt_matrix(tilt, xrot).tolist()),
                    m.Affine(name=IN_PLANE_ROTATION, affine=in_plane_matrix(rot).tolist()),
                    m.Translation(name=SHIFT, translation=[float(t_cets[0]), float(t_cets[1])]),
                ],
            ),
        )
    return m.Alignment(tilt_series_id=tilt_series_id, projection_alignments=projections)


# --------------------------------------------------------------------------- CETS -> hub


def _ttype(t) -> str:
    v = getattr(t, "transformation_type", None)
    return str(getattr(v, "value", v) or "")


def fold_projection(pa) -> Tuple[np.ndarray, np.ndarray]:
    """Fold a ``ProjectionAlignment.sequence`` into ``(R [3x3], t [2])`` generically: every ``Affine`` is
    applied as ``M <- A·M, v <- A·v`` and every ``Translation`` as ``v <- v + tau`` in list order, so the
    3-entry profile form, the 2-entry ``[Affine, Translation]`` form and reordered names all fold to the
    same operator. Affines must be 3×3 proper rotations (2D-homogeneous affines with a translation column are
    rejected); a translation may have 2 or 3 components (z ignored).
    """
    name = getattr(pa, "name", None)
    if name != TOMOGRAM_TO_PROJECTION:
        raise ValueError(
            f"projection alignment {getattr(pa, 'id', '?')!r} is named {name!r}, not {TOMOGRAM_TO_PROJECTION!r}: "
            "not a cets-rigid document (use an adapter for other encodings)",
        )
    if pa.input != PHYSICAL_CS or pa.output != PHYSICAL_CS:
        raise ValueError(
            f"projection alignment {pa.id!r} must map physical -> physical, got {pa.input!r} -> {pa.output!r}",
        )
    r = np.eye(3)
    v = np.zeros(3)
    steps = list(pa.sequence or [])
    if not steps:
        raise ValueError(f"projection alignment {pa.id!r} has an empty sequence")
    for step in steps:
        kind = _ttype(step)
        if kind == "affine":
            a = np.array(step.affine, dtype=np.float64)
            check_rotation(a, f"projection alignment {pa.id!r} affine {getattr(step, 'name', '?')!r}")
            r = a @ r
            v = a @ v
        elif kind == "translation":
            tau = np.array(step.translation, dtype=np.float64)
            if tau.shape[0] not in (2, 3):
                raise ValueError(f"projection alignment {pa.id!r}: translation has {tau.shape[0]} components")
            v[: tau.shape[0]] += tau
        else:
            raise ValueError(f"projection alignment {pa.id!r}: unsupported step {kind!r}")
    return r, v[:2]


def alignment_from_cets(
    cets_alignment,
    *,
    tilt_series,
    reference: ReferenceVolume,
    target_frame: FrameConvention,
    native_dimension_a: Optional[Dict[str, float]] = None,
    format_: str = "CETS",
    is_portal_standard: bool = True,
) -> Alignment:
    """Decode a profile CETS ``Alignment`` into a hub ``Alignment`` expressed in ``target_frame``.

    Every ``ProjectionAlignment`` is bound to its tilt image by ``tilt_image_id`` (the image's ``section`` is
    the raw z index and its own ``array_to_physical`` gives the image frame). ``native_dimension_a`` is the
    native volume box to record in the hub (default: the reference tomogram's extent).
    """
    images = {im.id: im for im in (tilt_series.images or [])}
    if cets_alignment.tilt_series_id not in (None, tilt_series.id):
        raise ValueError(
            f"alignment is for tilt series {cets_alignment.tilt_series_id!r}, not {tilt_series.id!r}",
        )
    native = native_dimension_a or {
        "x": float(reference.extent_a[0]),
        "y": float(reference.extent_a[1]),
        "z": float(reference.extent_a[2]),
    }
    reconcile_volume(native, reference)

    params: List[PerSectionAlignmentParameters] = []
    for pa in cets_alignment.projection_alignments or []:
        if pa.tilt_image_id is None or pa.tilt_image_id not in images:
            raise ValueError(
                f"projection alignment {pa.id!r}: tilt_image_id {pa.tilt_image_id!r} not in tilt series {tilt_series.id!r}",
            )
        im = images[pa.tilt_image_id]
        if im.section is None:
            raise ValueError(f"tilt image {im.id!r} has no section index")
        fr = image_frame(im)
        s = fr.isotropic_spacing
        r, t_cets = fold_projection(pa)
        rot, tilt, xrot = decompose(r)
        d_img, d_vol = _deltas(target_frame, fr, reference, native)
        t_native = t_cets - d_img + (r @ d_vol)[:2]
        params.append(
            PerSectionAlignmentParameters(
                z_index=int(im.section),
                tilt_angle=tilt,
                volume_x_rotation=xrot,
                in_plane_rotation=ang2mat(rot).tolist(),
                x_offset=float(t_native[0] / s),
                y_offset=float(t_native[1] / s),
            ),
        )
    params.sort(key=lambda p: p.z_index)
    if len({p.z_index for p in params}) != len(params):
        raise ValueError("several projection alignments reference the same tilt image")

    return Alignment(
        affine_transformation_matrix=np.eye(4).tolist(),
        alignment_type="GLOBAL",
        format=format_,
        is_portal_standard=is_portal_standard,
        tilt_offset=0.0,
        volume_offset={"x": 0.0, "y": 0.0, "z": 0.0},
        x_rotation_offset=0.0,
        per_section_alignment_parameters=params,
        volume_dimension={k: float(v) for k, v in native.items()},
    )


def project_points(cets_alignment, points_tomo_a: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate the CETS chain: centred tomogram Å points ``(N,3)`` -> centred image Å ``(N,2)`` per
    projection alignment id (test/debug helper that only uses the document)."""
    out = {}
    for pa in cets_alignment.projection_alignments or []:
        r, t = fold_projection(pa)
        out[pa.id] = (np.asarray(points_tomo_a, dtype=np.float64) @ r.T)[:, :2] + t
    return out


def select_alignment(region, selector: Optional[object] = None, tilt_series_id: Optional[str] = None):
    """Pick one ``Alignment`` of a region: by index, by projection-id name prefix, or automatically when
    exactly one candidate exists; ambiguity is an error, never a guess."""
    candidates = [a for a in (region.alignments or []) if tilt_series_id is None or a.tilt_series_id == tilt_series_id]
    if not candidates:
        raise ValueError(
            f"region {region.id!r} has no alignment"
            + (f" for tilt series {tilt_series_id!r}" if tilt_series_id else ""),
        )
    if selector is None:
        if len(candidates) == 1:
            return candidates[0]
        names = [alignment_name_of(a) for a in candidates]
        raise ValueError(f"region {region.id!r} has {len(candidates)} alignments {names}; select one with --alignment")
    if isinstance(selector, int):
        return candidates[selector]
    matches = [a for a in candidates if alignment_name_of(a) == selector]
    if len(matches) != 1:
        raise ValueError(
            f"alignment {selector!r} matches {len(matches)} of {[alignment_name_of(a) for a in candidates]}",
        )
    return matches[0]


def alignment_name_of(cets_alignment) -> Optional[str]:
    """The alignment-instance name encoded in the projection-alignment ids (``<ts>_<name>_align_<z>``)."""
    pas = cets_alignment.projection_alignments or []
    if not pas or cets_alignment.tilt_series_id is None:
        return None
    pid = pas[0].id
    prefix = f"{cets_alignment.tilt_series_id}_"
    if not pid.startswith(prefix) or "_align_" not in pid:
        return None
    return pid[len(prefix) :].rsplit("_align_", 1)[0]


def tomogram_ids_for(region, cets_alignment, companion=None) -> List[str]:
    """Tomogram ids bound to an alignment: from the companion when present, else every tomogram of the
    alignment's tilt series (the caller must select when there are several)."""
    if companion is not None:
        name = alignment_name_of(cets_alignment)
        entry = companion.alignment(cets_alignment.tilt_series_id, name) if name else None
        if entry is not None and entry.tomogram_ids:
            return list(entry.tomogram_ids)
    return [t.id for t in (region.tomograms or []) if t.tilt_series_id == cets_alignment.tilt_series_id]


def select_tomogram(region, cets_alignment, selector: Optional[str] = None, companion=None):
    """The reference tomogram: explicit selector > companion ``reference_tomogram_id`` > the single candidate."""
    if selector is not None:
        return find_by_id(region.tomograms, selector, "tomogram")
    if companion is not None:
        name = alignment_name_of(cets_alignment)
        entry = companion.alignment(cets_alignment.tilt_series_id, name) if name else None
        if entry is not None and entry.reference_tomogram_id:
            return find_by_id(region.tomograms, entry.reference_tomogram_id, "tomogram")
    ids = tomogram_ids_for(region, cets_alignment, companion)
    if len(ids) == 1:
        return find_by_id(region.tomograms, ids[0], "tomogram")
    raise ValueError(
        f"alignment {alignment_name_of(cets_alignment)!r} of region {region.id!r} has {len(ids)} candidate tomograms {ids}; "
        "select one with --tomogram",
    )


__all__ = [
    "CETS_FRAME",
    "ReferenceVolume",
    "alignment_from_cets",
    "alignment_name_of",
    "alignment_to_cets",
    "fold_projection",
    "project_points",
    "reconcile_volume",
    "select_alignment",
    "select_tomogram",
    "tomogram_ids_for",
]
