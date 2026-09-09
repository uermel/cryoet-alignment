"""Coordinate frames of the ``cets-rigid/0.1`` profile: the ``array``/``physical`` coordinate systems, the
``array_to_physical`` transform, and the centre conventions that relate native tool frames to CETS.

CETS (issue #1 / PR #34): ``array`` = integer indices, corner origin; ``physical`` = Å with the origin at the
array index ``floor(N/2)`` (``center_array_origin`` translation of ``-floor(N/2)`` followed by the pixel
scale). Native tools differ (``FrameConvention``): AreTomo3 and Warp centre their image and volume frames at
``N/2`` (float; ``arewarpion/frames.py``, ``TiltSeries.cs:407-408``), RELION's projection matrix at
``floor(N/2)`` (``tomogram.cpp:44,53``). For even sizes these agree; for odd sizes they differ by half a pixel
and the codec corrects for it on BOTH directions with ``delta = native_centre - cets_centre`` (Å).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np

from cryoet_alignment.io.cets.profile import (
    ARRAY_CS,
    ARRAY_TO_PHYSICAL,
    CENTER_ARRAY_ORIGIN,
    PHYSICAL_CS,
    SCALE_TO_PHYSICAL,
    UNIT_ANGSTROM,
    UNIT_PIXEL,
    UNIT_VOXEL,
    require_cets,
)

CenterKind = Literal["half", "floor"]


@dataclass(frozen=True)
class FrameConvention:
    """Where a native tool puts the origin of its centred image / volume frame, in array index units."""

    image_center: CenterKind
    volume_center: CenterKind

    def centre_px(self, size: Sequence[int], which: str) -> np.ndarray:
        kind = self.image_center if which == "image" else self.volume_center
        return np.array([centre_index(int(n), kind) for n in size], dtype=np.float64)


#: Native conventions of the hub sources (``Alignment.format``).
FRAME_CONVENTIONS = {
    "ARETOMO3": FrameConvention("half", "half"),
    "WARP": FrameConvention("half", "half"),
    "RELION": FrameConvention("floor", "floor"),
}
#: The CETS physical frame itself.
CETS_FRAME = FrameConvention("floor", "floor")


def centre_index(n: int, kind: CenterKind) -> float:
    return n / 2.0 if kind == "half" else float(n // 2)


def frame_convention_for(fmt: str) -> FrameConvention:
    """The frame convention of a hub ``Alignment.format``; IMOD is deliberately unpinned."""
    try:
        return FRAME_CONVENTIONS[fmt.upper()]
    except KeyError as exc:
        raise ValueError(
            f"no pinned centre convention for alignment format {fmt!r} (known: {sorted(FRAME_CONVENTIONS)}); "
            "pass frame=FrameConvention(...) explicitly",
        ) from exc


# --------------------------------------------------------------------------- coordinate systems


def array_cs(ndim: int):
    m = require_cets()
    unit = UNIT_VOXEL if ndim == 3 else UNIT_PIXEL
    return m.CoordinateSystem(
        name=ARRAY_CS,
        axes=[m.Axis(name=n, axis_unit=unit, axis_type=m.AxisType.array) for n in "xyz"[:ndim]],
    )


def physical_cs(ndim: int):
    m = require_cets()
    return m.CoordinateSystem(
        name=PHYSICAL_CS,
        axes=[m.Axis(name=n, axis_unit=UNIT_ANGSTROM, axis_type=m.AxisType.space) for n in "xyz"[:ndim]],
    )


def array_to_physical(size_px: Sequence[int], spacing_a: Sequence[float]):
    """``Sequence[Translation(-floor(N/2)), Scale(s)]`` named ``array_to_physical``."""
    m = require_cets()
    ndim = len(size_px)
    if len(spacing_a) != ndim:
        raise ValueError(f"spacing has {len(spacing_a)} values for {ndim} axes")
    return m.Sequence(
        name=ARRAY_TO_PHYSICAL,
        input=ARRAY_CS,
        output=PHYSICAL_CS,
        sequence=[
            m.Translation(name=CENTER_ARRAY_ORIGIN, translation=[-float(int(n) // 2) for n in size_px]),
            m.Scale(name=SCALE_TO_PHYSICAL, scale=[float(s) for s in spacing_a]),
        ],
    )


def attach_frames(image, size_px: Sequence[int], spacing_a: Sequence[float]) -> None:
    """Give a CETS image entity the profile's ``array``/``physical`` systems and ``array_to_physical``."""
    ndim = len(size_px)
    image.coordinate_systems = [array_cs(ndim), physical_cs(ndim)]
    image.coordinate_transformations = [array_to_physical(size_px, spacing_a)]


# --------------------------------------------------------------------------- reading frames back


@dataclass(frozen=True)
class ImageFrame:
    """What an image's ``array_to_physical`` chain says: size, spacing and the array index of the
    physical origin per axis (``corner_origin`` when the chain is a bare ``Scale``)."""

    size_px: tuple
    spacing_a: tuple
    origin_index: tuple  # physical origin as an array index, per axis

    @property
    def ndim(self) -> int:
        return len(self.size_px)

    @property
    def isotropic_spacing(self) -> float:
        s = self.spacing_a
        if max(s) - min(s) > 1e-9 * max(s):
            raise ValueError(f"anisotropic spacing {s} is not supported by the rigid profile")
        return float(s[0])

    def centre_delta_a(self, native: CenterKind) -> np.ndarray:
        """``native_centre - cets_centre`` in Å per axis."""
        return np.array(
            [(centre_index(int(n), native) - o) * s for n, o, s in zip(self.size_px, self.origin_index, self.spacing_a)],
            dtype=np.float64,
        )


def _ttype(t) -> str:
    v = getattr(t, "transformation_type", None)
    return str(getattr(v, "value", v) or "")


def _size_of(image) -> tuple:
    dims = [getattr(image, "width", None), getattr(image, "height", None), getattr(image, "depth", None)]
    dims = [d for d in dims if d is not None]
    if len(dims) not in (2, 3):
        raise ValueError(f"image {getattr(image, 'id', '?')!r} has no width/height(/depth)")
    return tuple(int(d) for d in dims)


def image_frame(image) -> ImageFrame:
    """Validate an entity's coordinate chain against the profile and return its ``ImageFrame``.

    Accepted: ``array`` (axis_type array) + ``physical`` (axis_type space, unit angstrom) coordinate
    systems and exactly one ``array_to_physical`` transform that is either
    ``Sequence[Translation, Scale]`` (origin at ``-translation``) or a bare ``Scale`` (corner origin —
    never an implied centering). Everything else is rejected naming the offending element.
    """
    size = _size_of(image)
    ndim = len(size)
    what = f"image {getattr(image, 'id', None) or getattr(image, 'path', '?')!r}"

    systems = {cs.name: cs for cs in (getattr(image, "coordinate_systems", None) or [])}
    for name, want_type in ((ARRAY_CS, "array"), (PHYSICAL_CS, "space")):
        cs = systems.get(name)
        if cs is None:
            raise ValueError(f"{what}: missing coordinate system {name!r}")
        if len(cs.axes) != ndim:
            raise ValueError(f"{what}: coordinate system {name!r} has {len(cs.axes)} axes, image has {ndim}")
        for ax in cs.axes:
            atype = str(getattr(ax.axis_type, "value", ax.axis_type))
            if atype != want_type:
                raise ValueError(f"{what}: axis {ax.name!r} of {name!r} has axis_type {atype!r}, expected {want_type!r}")
            if name == PHYSICAL_CS and str(getattr(ax.axis_unit, "value", ax.axis_unit)) != UNIT_ANGSTROM:
                raise ValueError(f"{what}: physical axis {ax.name!r} unit is not angstrom")

    chains = [t for t in (getattr(image, "coordinate_transformations", None) or []) if getattr(t, "name", None) == ARRAY_TO_PHYSICAL]
    if len(chains) != 1:
        raise ValueError(f"{what}: expected exactly one transform named {ARRAY_TO_PHYSICAL!r}, found {len(chains)}")
    chain = chains[0]
    if chain.input != ARRAY_CS or chain.output != PHYSICAL_CS:
        raise ValueError(f"{what}: {ARRAY_TO_PHYSICAL!r} must map {ARRAY_CS!r} -> {PHYSICAL_CS!r}, got {chain.input!r} -> {chain.output!r}")

    kind = _ttype(chain)
    if kind == "scale":
        scale = _broadcast(list(chain.scale), ndim, what, "scale")
        return ImageFrame(size_px=size, spacing_a=tuple(scale), origin_index=tuple([0.0] * ndim))
    if kind != "sequence":
        raise ValueError(f"{what}: {ARRAY_TO_PHYSICAL!r} is a {kind!r}; the profile allows a Scale or Sequence[Translation, Scale]")
    steps = list(chain.sequence or [])
    if len(steps) != 2 or _ttype(steps[0]) != "translation" or _ttype(steps[1]) != "scale":
        raise ValueError(f"{what}: {ARRAY_TO_PHYSICAL!r} must be Sequence[Translation, Scale], got {[_ttype(s) for s in steps]}")
    trans = _broadcast(list(steps[0].translation), ndim, what, "translation")
    scale = _broadcast(list(steps[1].scale), ndim, what, "scale")
    if any(v <= 0 for v in scale):
        raise ValueError(f"{what}: non-positive scale {scale}")
    return ImageFrame(size_px=size, spacing_a=tuple(scale), origin_index=tuple(-v for v in trans))


def _broadcast(values: List[float], ndim: int, what: str, label: str) -> List[float]:
    if len(values) == 1:
        return [float(values[0])] * ndim
    if len(values) != ndim:
        raise ValueError(f"{what}: {label} has {len(values)} values for {ndim} axes")
    return [float(v) for v in values]


def pixel_size_of(image) -> float:
    """Isotropic spacing (Å) of an entity's ``array_to_physical``."""
    return image_frame(image).isotropic_spacing


def find_by_id(items: Optional[list], id_: str, what: str):
    for it in items or []:
        if getattr(it, "id", None) == id_:
            return it
    raise ValueError(f"{what} {id_!r} not found")
