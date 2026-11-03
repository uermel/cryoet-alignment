"""Annotations of the rigid profile: point sets, oriented particles and segmentation masks bound to a tomogram.

Contract (``docs/cets.md`` "Annotations"): an annotation binds to exactly one tomogram entity via
``source_tomogram_id``; its coordinates are Å in that tomogram's centred ``physical`` frame (origin at array
index ``floor(N/2)``, corner-anchored voxel grid, z = tomogram z), declared by a ``physical`` coordinate system
and one transform named ``annotation_to_tomogram`` (``physical`` → ``physical``; the identity when written by
this codec). ``PointMatrixSet3D.matrix3D[i]`` is the active rotation mapping particle/reference-map coordinates
to tomogram coordinates (``p_tomo = M · p_map + origin``; RELION ``A(rot, tilt, psi)``, Warp ``Matrix3.Euler``,
see ``euler.py``). Masks carry their own ``array``/``physical`` frame like any image.

Readers fold any ``annotation_to_tomogram`` chain of Identity / Translation / Scale / Affine / Sequence into
``(A, t)`` (the pattern of ``alignment.fold_projection``) and apply ``p_tomo = A · p + t``; other names or
systems are refused naming the element. Grid offsets between reconstruction engines are not modelled.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cryoet_alignment.io.cets.frames import ImageFrame, attach_frames, find_by_id, image_frame, physical_cs
from cryoet_alignment.io.cets.profile import ANNOTATION_TO_TOMOGRAM, PHYSICAL_CS, require_cets
from cryoet_alignment.io.cets.rotation import check_rotation

__all__ = [
    "TomogramFrame",
    "tomogram_frame",
    "point_set_entity",
    "mask_entity",
    "fold_annotation_transform",
    "ResolvedPoints",
    "annotation_points",
    "annotation_kind",
    "select_annotations",
    "annotation_id",
    "POINT_KINDS",
]

POINT_KINDS = ("points", "oriented_points")


# --------------------------------------------------------------------------- tomogram frame helpers


@dataclass(frozen=True)
class TomogramFrame:
    """The corner and centre bookkeeping of one tomogram entity (Å, per axis)."""

    tomogram_id: str
    frame: ImageFrame

    @property
    def spacing_a(self) -> float:
        return self.frame.isotropic_spacing

    @property
    def size_px(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.frame.size_px)  # type: ignore[return-value]

    @property
    def corner_a(self) -> np.ndarray:
        """``c``: the CETS origin measured from the corner of the voxel grid = ``origin_index · s``."""
        return np.array(self.frame.origin_index, dtype=np.float64) * np.array(self.frame.spacing_a)

    @property
    def extent_a(self) -> np.ndarray:
        return np.array(self.frame.size_px, dtype=np.float64) * np.array(self.frame.spacing_a)

    @property
    def float_centre_delta_a(self) -> np.ndarray:
        """``h = E/2 − c``: RELION's float ``N/2`` centre minus the CETS centre (0 or ``s/2`` per axis)."""
        return self.extent_a / 2.0 - self.corner_a

    def corner_to_cets(self, points_corner_a) -> np.ndarray:
        return np.asarray(points_corner_a, dtype=np.float64).reshape(-1, 3) - self.corner_a

    def cets_to_corner(self, points_a) -> np.ndarray:
        return np.asarray(points_a, dtype=np.float64).reshape(-1, 3) + self.corner_a

    def inside(self, points_corner_a) -> np.ndarray:
        """Boolean mask: point inside ``[0, E)`` per axis (corner-anchored Å)."""
        p = np.asarray(points_corner_a, dtype=np.float64).reshape(-1, 3)
        return np.all((p >= 0.0) & (p < self.extent_a), axis=1)


def tomogram_frame(tomogram) -> TomogramFrame:
    fr = image_frame(tomogram)
    if fr.ndim != 3:
        raise ValueError(f"tomogram {getattr(tomogram, 'id', '?')!r} is not a 3D image")
    return TomogramFrame(tomogram_id=str(tomogram.id), frame=fr)


# --------------------------------------------------------------------------- builders


def _identity_to_tomogram():
    m = require_cets()
    return m.Identity(name=ANNOTATION_TO_TOMOGRAM, input=PHYSICAL_CS, output=PHYSICAL_CS)


def annotation_id(tomogram_id: str, key: str) -> str:
    """``<tomogram_id>_ann_<key>``."""
    return f"{tomogram_id}_ann_{key}"


def point_set_entity(
    *,
    annotation_id: str,
    tomogram_id: str,
    points_a,
    matrices=None,
    name: Optional[str] = None,
):
    """``PointSet3D`` (``matrices is None``) or ``PointMatrixSet3D`` in the tomogram's centred physical frame."""
    m = require_cets()
    pts = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError(f"annotation {annotation_id!r}: no points")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"annotation {annotation_id!r}: non-finite coordinates")
    common = dict(
        id=annotation_id,
        name=name,
        source_tomogram_id=tomogram_id,
        coordinate_systems=[physical_cs(3)],
        coordinate_transformations=[_identity_to_tomogram()],
        origin3D=[[float(v) for v in row] for row in pts],
    )
    if matrices is None:
        return m.PointSet3D(**common)
    mats = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    if len(mats) != len(pts):
        raise ValueError(f"annotation {annotation_id!r}: {len(mats)} matrices for {len(pts)} points")
    for i, r in enumerate(mats):
        check_rotation(r, f"annotation {annotation_id!r} matrix3D[{i}]")
    return m.PointMatrixSet3D(matrix3D=[[[float(v) for v in row] for row in r] for r in mats], **common)


def mask_entity(
    *,
    annotation_id: str,
    tomogram_id: str,
    path: Optional[str],
    size_px: Sequence[int],
    voxel_size_a: float,
    name: Optional[str] = None,
):
    """``SegmentationMask3D`` with its own ``array``/``physical`` frame and an identity ``annotation_to_tomogram``."""
    m = require_cets()
    size = tuple(int(v) for v in size_px)
    if len(size) != 3 or any(v <= 0 for v in size):
        raise ValueError(f"annotation {annotation_id!r}: mask size {size_px} is not a positive 3D shape")
    mask = m.SegmentationMask3D(
        id=annotation_id,
        name=name,
        source_tomogram_id=tomogram_id,
        path=path,
        width=size[0],
        height=size[1],
        depth=size[2],
    )
    attach_frames(mask, size, (float(voxel_size_a),) * 3)
    mask.coordinate_transformations = list(mask.coordinate_transformations or []) + [_identity_to_tomogram()]
    return mask


# --------------------------------------------------------------------------- reading


def _ttype(t) -> str:
    v = getattr(t, "transformation_type", None)
    return str(getattr(v, "value", v))


def _fold_steps(steps, what: str) -> Tuple[np.ndarray, np.ndarray]:
    a = np.eye(3)
    t = np.zeros(3)
    for step in steps:
        kind = _ttype(step)
        if kind == "identity":
            continue
        if kind == "translation":
            tau = np.asarray(step.translation, dtype=np.float64)
            if tau.shape != (3,):
                raise ValueError(f"{what}: translation has {tau.shape[0]} components, expected 3")
            t = t + tau
        elif kind == "scale":
            s = np.asarray(step.scale, dtype=np.float64)
            if s.shape != (3,):
                raise ValueError(f"{what}: scale has {s.shape[0]} components, expected 3")
            a = np.diag(s) @ a
            t = s * t
        elif kind == "affine":
            mat = np.asarray(step.affine, dtype=np.float64)
            if mat.shape != (3, 3):
                raise ValueError(f"{what}: affine has shape {mat.shape}, expected 3x3")
            a = mat @ a
            t = mat @ t
        elif kind == "sequence":
            a2, t2 = _fold_steps(list(step.sequence or []), what)
            a = a2 @ a
            t = a2 @ t + t2
        else:
            raise ValueError(f"{what}: unsupported step {kind!r} ({getattr(step, 'name', '?')!r})")
    return a, t


def fold_annotation_transform(annotation) -> Tuple[np.ndarray, np.ndarray]:
    """Fold the annotation's ``annotation_to_tomogram`` chain into ``(A, t)`` so ``p_tomo = A·p + t`` (Å)."""
    what = f"annotation {getattr(annotation, 'id', '?')!r}"
    systems = {cs.name: cs for cs in (getattr(annotation, "coordinate_systems", None) or [])}
    cs = systems.get(PHYSICAL_CS)
    if cs is None:
        raise ValueError(f"{what}: missing coordinate system {PHYSICAL_CS!r}")
    for ax in cs.axes:
        unit = str(getattr(ax.axis_unit, "value", ax.axis_unit))
        if unit != "angstrom":
            raise ValueError(f"{what}: physical axis {ax.name!r} unit is {unit!r}, expected 'angstrom'")
    chains = [
        tr
        for tr in (getattr(annotation, "coordinate_transformations", None) or [])
        if getattr(tr, "name", None) == ANNOTATION_TO_TOMOGRAM
    ]
    if len(chains) != 1:
        raise ValueError(
            f"{what}: expected exactly one transform named {ANNOTATION_TO_TOMOGRAM!r}, found {len(chains)}",
        )
    chain = chains[0]
    if chain.input != PHYSICAL_CS or chain.output != PHYSICAL_CS:
        raise ValueError(
            f"{what}: {ANNOTATION_TO_TOMOGRAM!r} must map {PHYSICAL_CS!r} -> {PHYSICAL_CS!r}, "
            f"got {chain.input!r} -> {chain.output!r}",
        )
    return _fold_steps([chain], what)


def _rotational_part(a: np.ndarray, what: str) -> np.ndarray:
    """The rotation of a folded ``annotation_to_tomogram`` affine (uniform scale allowed, no reflection)."""
    u, sv, vt = np.linalg.svd(a)
    if sv.max() - sv.min() > 1e-9 * sv.max():
        raise ValueError(f"{what}: {ANNOTATION_TO_TOMOGRAM!r} scales anisotropically; orientations cannot be carried")
    r = u @ vt
    if np.linalg.det(r) < 0:
        raise ValueError(f"{what}: {ANNOTATION_TO_TOMOGRAM!r} contains a reflection; orientations cannot be carried")
    check_rotation(r, f"{what} {ANNOTATION_TO_TOMOGRAM!r} rotation")
    return r


@dataclass
class ResolvedPoints:
    """Points of an annotation in its tomogram's centred physical frame (Å)."""

    annotation_id: str
    tomogram: object
    frame: TomogramFrame
    points_a: np.ndarray  # (N,3)
    matrices: Optional[np.ndarray]  # (N,3,3) or None

    @property
    def n(self) -> int:
        return int(len(self.points_a))

    @property
    def points_corner_a(self) -> np.ndarray:
        return self.frame.cets_to_corner(self.points_a)


def annotation_kind(annotation) -> str:
    v = getattr(annotation, "annotation_type", None)
    kind = str(getattr(v, "value", v))
    return {
        "point_set_3D": "points",
        "point_matrix_set_3D": "oriented_points",
        "segmentation_mask_3D": "mask",
    }.get(kind, kind)


def annotation_points(annotation, region, *, tomogram=None) -> ResolvedPoints:
    """Resolve a ``PointSet3D`` / ``PointMatrixSet3D`` against its tomogram and apply ``annotation_to_tomogram``."""
    what = f"annotation {getattr(annotation, 'id', '?')!r}"
    kind = annotation_kind(annotation)
    if kind not in POINT_KINDS:
        raise ValueError(f"{what}: {kind!r} is not a point annotation")
    tid = getattr(annotation, "source_tomogram_id", None)
    if tomogram is None:
        if not tid:
            raise ValueError(f"{what}: no source_tomogram_id; pass the tomogram explicitly")
        tomogram = find_by_id(region.tomograms, tid, "tomogram")
    elif tid and str(tomogram.id) != str(tid):
        raise ValueError(f"{what}: bound to tomogram {tid!r}, not {tomogram.id!r}")
    frame = tomogram_frame(tomogram)
    a, t = fold_annotation_transform(annotation)
    pts = np.asarray(annotation.origin3D or [], dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError(f"{what}: no points")
    pts = pts @ a.T + t
    mats = None
    if kind == "oriented_points":
        raw = np.asarray(annotation.matrix3D or [], dtype=np.float64).reshape(-1, 3, 3)
        if len(raw) != len(pts):
            raise ValueError(f"{what}: {len(raw)} matrices for {len(pts)} points")
        for i, r in enumerate(raw):
            check_rotation(r, f"{what} matrix3D[{i}]")
        if not np.allclose(a, np.eye(3)):
            raw = np.einsum("ij,njk->nik", _rotational_part(a, what), raw)
        mats = raw
    return ResolvedPoints(annotation_id=str(annotation.id), tomogram=tomogram, frame=frame, points_a=pts, matrices=mats)


def select_annotations(
    region,
    *,
    ids: Optional[Sequence[str]] = None,
    tomogram_id: Optional[str] = None,
    kinds: Optional[Sequence[str]] = None,
) -> List[object]:
    """Annotations of a region filtered by id, bound tomogram and kind (``points``, ``oriented_points``, ``mask``)."""
    out = []
    wanted = set(ids or [])
    for ann in region.annotations or []:
        if wanted and str(ann.id) not in wanted:
            continue
        if tomogram_id is not None and getattr(ann, "source_tomogram_id", None) != tomogram_id:
            continue
        if kinds is not None and annotation_kind(ann) not in kinds:
            continue
        out.append(ann)
    if wanted:
        missing = wanted - {str(a.id) for a in out}
        if missing:
            raise ValueError(f"annotation(s) not in region {region.id!r}: {sorted(missing)}")
    return out
