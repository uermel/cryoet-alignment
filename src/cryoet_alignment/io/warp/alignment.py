"""Warp tilt-series XML — the RIGID subset, read and written natively (no torch/warpylib).

What is modelled (mirrors Warp's ``<TiltSeries>`` metadata file, ``WarpLib/TiltSeries/TiltSeries.cs``):

* root attributes ``ImageDimensionsAngstrom`` / ``VolumeDimensionsAngstrom`` (Å), ``LevelAngleX`` /
  ``LevelAngleY`` (degrees), ``AreAnglesInverted``, ``DataDirectory``;
* per-tilt elements (one line per tilt, newline separated): ``Angles`` (defines the tilt count),
  ``Dose``, ``UseTilt``, ``AxisAngle``, ``AxisOffsetX`` / ``AxisOffsetY`` (Å), ``MoviePath`` (blank rows
  are preserved);
* ``<CTF>`` ``Param`` entries (``PixelSize`` = the pixel size Warp used for CTF fitting, ``Voltage``,
  ``Cs``, ``Amplitude``, … — kept verbatim for round trips);
* the per-tilt CTF grids ``GridCTF`` (defocus, µm), ``GridCTFDefocusDelta`` (µm), ``GridCTFDefocusAngle``
  (degrees), ``GridCTFPhase`` (multiples of π) when they are ``1×1×T`` (or ``1×1×1``);
* an AUDIT of every deformation grid (``GridMovementX/Y``, ``GridVolumeWarpX/Y/Z``, ``GridAngleX/Y/Z``):
  spatially constant grids are rigid contributions and are folded into the per-tilt values
  (``movement_x/y``, ``constant_volume_warp``); spatially varying grids are reported
  (``varying_grid_names``) so callers can refuse or explicitly drop them. A single grid node with a
  non-zero value is NOT "no deformation": Warp subtracts ``GridMovement`` after and adds ``GridVolumeWarp``
  before the rigid projection (``TiltSeries.cs:402-478``).

Warp positions a tilt image row by row against the ``.tomostar`` it was created from, so a written XML
carries exactly one row per tilt series row, dark tilts included (``UseTilt=False``).
"""

from typing import Dict, List, Optional
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from cryoet_alignment.io.base import PATH_TYPE, FileIOBase

# Grids Warp writes for a tilt series, in file order, with their default node value.
_CTF_GRIDS = ("GridCTF", "GridCTFDefocusDelta", "GridCTFDefocusAngle", "GridCTFPhase")
_MOVEMENT_GRIDS = ("GridMovementX", "GridMovementY")
_VOLUME_WARP_GRIDS = ("GridVolumeWarpX", "GridVolumeWarpY", "GridVolumeWarpZ")
_ANGLE_GRIDS = ("GridAngleX", "GridAngleY", "GridAngleZ")
_OTHER_GRIDS = {
    "GridDoseBfacs": 0.0,
    "GridDoseBfacsDelta": 0.0,
    "GridDoseBfacsAngle": 0.0,
    "GridDoseWeights": 1.0,
    "GridLocationBfacs": 0.0,
    "GridLocationWeights": 1.0,
}


class WarpAlignmentEntry(BaseModel):
    """Per-tilt data in Warp convention (one XML row).

    Attributes:
        z_index: Row index in the XML / ``.tomostar`` (0-based). This is NOT a raw-stack section
            index of any other tool; rows are matched across tools by stage angle.
        tilt_angle: ``Angles`` entry in degrees (Warp's sign convention).
        tilt_axis_angle: ``AxisAngle`` in degrees.
        tilt_axis_offset_x: ``AxisOffsetX`` in Å (image-space shift after rotation).
        tilt_axis_offset_y: ``AxisOffsetY`` in Å.
        use_tilt: ``UseTilt`` — False marks a tilt Warp ignores (dark / excluded).
        dose: ``Dose`` — accumulated dose before this image in e/Å².
        movie_path: ``MoviePath`` entry (may be empty).
        movement_x: Rigid contribution of a spatially constant ``GridMovementX`` at this tilt (Å);
            Warp SUBTRACTS it from the projected position. Zero for 1×1×1 zero grids.
        movement_y: Same for ``GridMovementY``.
        defocus_um: ``GridCTF`` node for this tilt (µm), None when the XML has no per-tilt CTF grid.
        defocus_delta_um: ``GridCTFDefocusDelta`` node (µm).
        defocus_angle_deg: ``GridCTFDefocusAngle`` node (degrees).
        phase_shift_pi: ``GridCTFPhase`` node (multiples of π).
    """

    z_index: int
    tilt_angle: float
    tilt_axis_angle: float
    tilt_axis_offset_x: float
    tilt_axis_offset_y: float
    use_tilt: bool = True
    dose: float = 0.0
    movie_path: str = ""
    movement_x: float = 0.0
    movement_y: float = 0.0
    defocus_um: Optional[float] = None
    defocus_delta_um: Optional[float] = None
    defocus_angle_deg: Optional[float] = None
    phase_shift_pi: Optional[float] = None

    @property
    def effective_offset_x(self) -> float:
        """The rigid image-space shift Warp applies: ``AxisOffsetX - GridMovementX`` (Å)."""
        return self.tilt_axis_offset_x - self.movement_x

    @property
    def effective_offset_y(self) -> float:
        return self.tilt_axis_offset_y - self.movement_y


class GridAudit(BaseModel):
    """What the deformation grids of an XML contain, in rigid terms."""

    constant_volume_warp: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    varying_grid_names: List[str] = Field(default_factory=list)
    angle_grid_names: List[str] = Field(default_factory=list)
    ctf_grid_spatial: bool = False

    @property
    def has_varying_grids(self) -> bool:
        return bool(self.varying_grid_names)

    @property
    def has_angle_grids(self) -> bool:
        return bool(self.angle_grid_names)


def _attr_floats(root: ElementTree.Element, name: str, count: int) -> Optional[List[float]]:
    raw = root.get(name)
    if raw is None or not raw.strip():
        return None
    values = [float(v) for v in raw.split(",")]
    if len(values) != count:
        raise ValueError(f"attribute {name} has {len(values)} values, expected {count}")
    return values


def _per_tilt_lines(root: ElementTree.Element, name: str, n_tilts: int) -> Optional[List[str]]:
    """Newline-separated per-tilt text element, blank rows preserved. ``None`` when absent."""
    elem = root.find(name)
    if elem is None or elem.text is None:
        return None
    lines = [ln.strip() for ln in elem.text.split("\n")]
    # Warp indents the closing tag; drop leading/trailing empties only while the count exceeds n_tilts
    while len(lines) > n_tilts and lines and lines[-1] == "":
        lines.pop()
    while len(lines) > n_tilts and lines and lines[0] == "":
        lines.pop(0)
    if len(lines) != n_tilts:
        raise ValueError(f"element {name} has {len(lines)} rows, expected {n_tilts} (from Angles)")
    return lines


def _per_tilt_floats(root: ElementTree.Element, name: str, n_tilts: int) -> List[float]:
    lines = _per_tilt_lines(root, name, n_tilts)
    if lines is None:
        return [0.0] * n_tilts
    if any(ln == "" for ln in lines):
        raise ValueError(f"element {name} has blank rows")
    return [float(v) for v in lines]


def _per_tilt_bools(root: ElementTree.Element, name: str, n_tilts: int) -> List[bool]:
    lines = _per_tilt_lines(root, name, n_tilts)
    if lines is None:
        return [True] * n_tilts
    out = []
    for ln in lines:
        if ln.lower() == "true":
            out.append(True)
        elif ln.lower() == "false":
            out.append(False)
        else:
            raise ValueError(f"element {name}: cannot parse boolean {ln!r}")
    return out


def _grid_nodes(elem: ElementTree.Element) -> Dict[tuple, float]:
    nodes = {}
    for node in elem.findall("Node"):
        key = (
            int(node.get("X", "0")),
            int(node.get("Y", "0")),
            int(node.get("Z", "0")),
            int(node.get("W", "0")),
        )
        nodes[key] = float(node.get("Value", "0"))
    return nodes


def _grid_dims(elem: ElementTree.Element) -> tuple:
    return (
        int(elem.get("Width", "1")),
        int(elem.get("Height", "1")),
        int(elem.get("Depth", "1")),
        int(elem.get("Duration", "1")),
    )


def _per_tilt_grid_values(root: ElementTree.Element, name: str, n_tilts: int):
    """Per-tilt values of a grid whose only variation is along the tilt axis.

    Returns ``(values, spatial)``: ``values`` is None when the element is absent; ``spatial`` is True
    when the grid varies in X/Y too (values are then the per-tilt mean over the spatial nodes).
    Raises when the temporal extent is neither 1 nor ``n_tilts`` (its values at the tilts would be
    spline interpolations this module does not evaluate).
    """
    elem = root.find(name)
    if elem is None:
        return None, False
    w, h, d, _ = _grid_dims(elem)
    nodes = _grid_nodes(elem)
    if d not in (1, n_tilts):
        raise ValueError(f"{name}: Depth {d} is neither 1 nor the tilt count {n_tilts}; cannot read per-tilt values")
    values = []
    for t in range(n_tilts):
        z = 0 if d == 1 else t
        col = [v for (x, y, zz, _w), v in nodes.items() if zz == z]
        if not col:
            raise ValueError(f"{name}: no nodes for tilt {t}")
        values.append(sum(col) / len(col))
    spatial = w * h > 1
    return values, spatial


def _audit_grids(root: ElementTree.Element, n_tilts: int):
    """Classify the deformation grids and fold the spatially constant ones.

    Returns ``(movement_x, movement_y, audit)`` with per-tilt movement lists.
    """
    audit = GridAudit()
    movement = {"GridMovementX": [0.0] * n_tilts, "GridMovementY": [0.0] * n_tilts}

    for name in _MOVEMENT_GRIDS:
        elem = root.find(name)
        if elem is None:
            continue
        w, h, d, _ = _grid_dims(elem)
        nodes = _grid_nodes(elem)
        if not nodes:
            continue
        if w * h == 1 and d in (1, n_tilts):
            movement[name] = [nodes.get((0, 0, 0 if d == 1 else t, 0), 0.0) for t in range(n_tilts)]
        elif len(set(nodes.values())) == 1:
            movement[name] = [next(iter(nodes.values()))] * n_tilts
        else:
            audit.varying_grid_names.append(name)

    warp = [0.0, 0.0, 0.0]
    for i, name in enumerate(_VOLUME_WARP_GRIDS):
        elem = root.find(name)
        if elem is None:
            continue
        nodes = _grid_nodes(elem)
        if not nodes:
            continue
        if len(set(nodes.values())) == 1:
            warp[i] = next(iter(nodes.values()))
        else:
            audit.varying_grid_names.append(name)
    audit.constant_volume_warp = warp

    for name in _ANGLE_GRIDS:
        elem = root.find(name)
        if elem is None:
            continue
        if any(v != 0.0 for v in _grid_nodes(elem).values()):
            audit.angle_grid_names.append(name)

    return movement["GridMovementX"], movement["GridMovementY"], audit


class WarpAlignment(FileIOBase):
    """Warp tilt-series metadata, rigid subset.

    Attributes:
        n_tilts: Number of XML rows (tilts, dark ones included).
        pixel_size_a: The tilt-IMAGE pixel size in Å/px used to convert ``AxisOffset`` Å to pixels
            (``Alignment.from_warp``). Supplied by the caller from the image header; when not given
            it falls back to ``ctf_pixel_size_a`` and ``pixel_size_source`` says so.
        pixel_size_source: ``"explicit"`` or ``"ctf"``.
        ctf_pixel_size_a: ``<CTF><Param Name="PixelSize">`` — the sampling Warp used for CTF fitting
            (``BinnedPixelSizeMean``), which need not equal the stored tilt-image pixel size.
        image_dimensions_physical: ``[width_a, height_a]`` in Å.
        volume_dimensions_physical: ``[x_a, y_a, z_a]`` in Å — Warp's native reconstruction box.
        level_angle_x: ``LevelAngleX`` (degrees; a rotation about the in-plane axis perpendicular to the tilt axis).
        level_angle_y: ``LevelAngleY`` (degrees; constant stage-tilt offset added to every angle).
        are_angles_inverted: ``AreAnglesInverted`` — affects only Warp's defocus/depth handedness, never
            the projected XY positions.
        data_directory: ``DataDirectory`` attribute as read.
        ctf_params: Every ``<CTF><Param>`` as ``{Name: Value}`` strings (round-trip provenance).
        grid_audit: What the deformation grids contained (see module docstring).
        entries: Per-tilt rows.
    """

    n_tilts: int
    pixel_size_a: float
    pixel_size_source: str = "explicit"
    ctf_pixel_size_a: Optional[float] = None
    image_dimensions_physical: List[float]
    volume_dimensions_physical: List[float]
    level_angle_x: float = 0.0
    level_angle_y: float = 0.0
    are_angles_inverted: bool = False
    data_directory: str = ""
    ctf_params: Dict[str, str] = Field(default_factory=dict)
    grid_audit: GridAudit = Field(default_factory=GridAudit)
    entries: List[WarpAlignmentEntry]

    @property
    def has_ctf(self) -> bool:
        return all(e.defocus_um is not None for e in self.entries)

    @property
    def is_rigid(self) -> bool:
        """True when no spatially varying deformation grid is present."""
        return not self.grid_audit.has_varying_grids

    @classmethod
    def from_string(
        cls,
        text: str,
        pixel_size_a: Optional[float] = None,
        image_dims_a: Optional[List[float]] = None,
        volume_dims_a: Optional[List[float]] = None,
        strict_dims: bool = True,
    ) -> "WarpAlignment":
        """Parse a Warp tilt-series XML.

        Args:
            text: XML content (a leading UTF-8 BOM, as written by Warp, is tolerated).
            pixel_size_a: Tilt-image pixel size in Å/px (from the image header). Falls back to the
                ``<CTF>`` PixelSize with ``pixel_size_source="ctf"``; refused when neither exists.
            image_dims_a: Override for ``ImageDimensionsAngstrom`` (re-saved Warp XMLs carry "0, 0").
            volume_dims_a: Override for ``VolumeDimensionsAngstrom``.
            strict_dims: Refuse missing/zero dimensions unless overridden (default). ``False`` reads
                them as zeros, for inspection only.
        """
        root = ElementTree.fromstring(text.lstrip("﻿"))
        if root.tag != "TiltSeries":
            raise ValueError(f"root element is <{root.tag}>, expected <TiltSeries>")

        angles_elem = root.find("Angles")
        if angles_elem is None or not (angles_elem.text or "").strip():
            raise ValueError("no <Angles> element — not a Warp tilt-series XML")
        angles = [float(v) for v in angles_elem.text.split() if v.strip()]
        n_tilts = len(angles)

        image_dims = image_dims_a if image_dims_a is not None else _attr_floats(root, "ImageDimensionsAngstrom", 2)
        volume_dims = (
            volume_dims_a if volume_dims_a is not None else _attr_floats(root, "VolumeDimensionsAngstrom", 3)
        )
        for label, dims, n in (("ImageDimensionsAngstrom", image_dims, 2), ("VolumeDimensionsAngstrom", volume_dims, 3)):
            if strict_dims and (dims is None or any(float(v) <= 0 for v in dims)):
                raise ValueError(
                    f"{label} is missing or zero in this XML (Warp re-saves them that way); pass "
                    f"{'image_dims_a' if n == 2 else 'volume_dims_a'} from the image header / .settings",
                )
        image_dims = [float(v) for v in (image_dims or [0.0, 0.0])]
        volume_dims = [float(v) for v in (volume_dims or [0.0, 0.0, 0.0])]

        ctf_params: Dict[str, str] = {}
        for param in root.findall("./CTF/Param"):
            if param.get("Name") is not None:
                ctf_params[param.get("Name")] = param.get("Value", "")
        ctf_pixel = float(ctf_params["PixelSize"]) if ctf_params.get("PixelSize") else None

        source = "explicit"
        if pixel_size_a is None:
            pixel_size_a = ctf_pixel
            source = "ctf"
        if pixel_size_a is None:
            raise ValueError(
                "Cannot derive pixel_size_a — the Warp XML carries no <CTF><Param Name='PixelSize'/> entry. "
                "Pass `pixel_size_a=<tilt image Å/px>` (from the image header).",
            )

        axis_angles = _per_tilt_floats(root, "AxisAngle", n_tilts)
        offsets_x = _per_tilt_floats(root, "AxisOffsetX", n_tilts)
        offsets_y = _per_tilt_floats(root, "AxisOffsetY", n_tilts)
        doses = _per_tilt_floats(root, "Dose", n_tilts)
        use_tilt = _per_tilt_bools(root, "UseTilt", n_tilts)
        movie_paths = _per_tilt_lines(root, "MoviePath", n_tilts) or [""] * n_tilts

        movement_x, movement_y, audit = _audit_grids(root, n_tilts)

        ctf_values = {}
        for name in _CTF_GRIDS:
            values, spatial = _per_tilt_grid_values(root, name, n_tilts)
            ctf_values[name] = values
            audit.ctf_grid_spatial = audit.ctf_grid_spatial or spatial
        # Warp's default grids are 1x1x1 zeros: a defocus of exactly 0 um on every tilt is "no CTF fit".
        has_ctf = all(ctf_values[n] is not None for n in _CTF_GRIDS) and any(v != 0.0 for v in ctf_values["GridCTF"])

        entries = [
            WarpAlignmentEntry(
                z_index=i,
                tilt_angle=angles[i],
                tilt_axis_angle=axis_angles[i],
                tilt_axis_offset_x=offsets_x[i],
                tilt_axis_offset_y=offsets_y[i],
                use_tilt=use_tilt[i],
                dose=doses[i],
                movie_path=movie_paths[i],
                movement_x=movement_x[i],
                movement_y=movement_y[i],
                defocus_um=ctf_values["GridCTF"][i] if has_ctf else None,
                defocus_delta_um=ctf_values["GridCTFDefocusDelta"][i] if has_ctf else None,
                defocus_angle_deg=ctf_values["GridCTFDefocusAngle"][i] if has_ctf else None,
                phase_shift_pi=ctf_values["GridCTFPhase"][i] if has_ctf else None,
            )
            for i in range(n_tilts)
        ]

        return cls(
            n_tilts=n_tilts,
            pixel_size_a=float(pixel_size_a),
            pixel_size_source=source,
            ctf_pixel_size_a=ctf_pixel,
            image_dimensions_physical=image_dims,
            volume_dimensions_physical=volume_dims,
            level_angle_x=float(root.get("LevelAngleX", "0") or 0.0),
            level_angle_y=float(root.get("LevelAngleY", "0") or 0.0),
            are_angles_inverted=(root.get("AreAnglesInverted", "False").strip().lower() == "true"),
            data_directory=root.get("DataDirectory", "") or "",
            ctf_params=ctf_params,
            grid_audit=audit,
            entries=entries,
        )

    @classmethod
    def from_file(
        cls,
        file_path: PATH_TYPE,
        pixel_size_a: Optional[float] = None,
        image_dims_a: Optional[List[float]] = None,
        volume_dims_a: Optional[List[float]] = None,
        strict_dims: bool = True,
    ) -> "WarpAlignment":
        """Load from a Warp tilt-series XML file (see ``from_string``)."""
        with open(file_path, "r") as file:
            return cls.from_string(
                file.read(),
                pixel_size_a=pixel_size_a,
                image_dims_a=image_dims_a,
                volume_dims_a=volume_dims_a,
                strict_dims=strict_dims,
            )

    # ------------------------------------------------------------------ writer

    @staticmethod
    def _grid(parent: ElementTree.Element, name: str, values: List[float], four_d: bool = False) -> None:
        """A ``1×1×len(values)`` cubic grid (or ``1×1×1×1`` linear grid) in Warp's node layout."""
        if four_d:
            g = ElementTree.SubElement(parent, name, Width="1", Height="1", Depth="1", Duration="1")
            ElementTree.SubElement(g, "Node", X="0", Y="0", Z="0", W="0", Value=f"{values[0]:.9g}")
            return
        g = ElementTree.SubElement(
            parent, name, Width="1", Height="1", Depth=str(len(values)), MarginX="0", MarginY="0", MarginZ="0",
        )
        for z, v in enumerate(values):
            ElementTree.SubElement(g, "Node", X="0", Y="0", Z=str(z), Value=f"{v:.9g}")

    def __str__(self) -> str:
        """Serialize to a complete rigid Warp tilt-series XML.

        Every element Warp/warpylib read is present: root attributes, the per-tilt elements (one row per
        entry, ``UseTilt``/``MoviePath`` included), the ``<CTF>`` params, per-tilt CTF grids (``1×1×T``
        when every entry carries CTF values, ``1×1×1`` zeros otherwise) and neutral ``1×1×1`` /
        ``1×1×1×1`` deformation grids. Folded ``movement_x/y`` are written back into ``AxisOffset``
        (``effective_offset``) with zero movement grids, so the projection is unchanged.
        """
        root = ElementTree.Element("TiltSeries")
        img = self.image_dimensions_physical
        vol = self.volume_dimensions_physical
        root.set("DataDirectory", self.data_directory)
        root.set("AreAnglesInverted", "True" if self.are_angles_inverted else "False")
        root.set("PlaneNormal", "0, 0, 0")
        root.set("LevelAngleX", f"{self.level_angle_x:.9g}")
        root.set("LevelAngleY", f"{self.level_angle_y:.9g}")
        root.set("Bfactor", "0")
        root.set("Weight", "1")
        root.set("MagnificationCorrection", "1, 0, 0, 1")
        root.set("ImageDimensionsAngstrom", f"{img[0]:.9g}, {img[1]:.9g}")
        root.set("VolumeDimensionsAngstrom", f"{vol[0]:.9g}, {vol[1]:.9g}, {vol[2]:.9g}")
        root.set("UnselectFilter", "False")
        root.set("UnselectManual", "")
        root.set("CTFResolutionEstimate", "0")

        per_tilt = {
            "Angles": [f"{e.tilt_angle:.9g}" for e in self.entries],
            "Dose": [f"{e.dose:.9g}" for e in self.entries],
            "UseTilt": ["True" if e.use_tilt else "False" for e in self.entries],
            "AxisAngle": [f"{e.tilt_axis_angle:.9g}" for e in self.entries],
            "AxisOffsetX": [f"{e.effective_offset_x:.9g}" for e in self.entries],
            "AxisOffsetY": [f"{e.effective_offset_y:.9g}" for e in self.entries],
            "MoviePath": [e.movie_path for e in self.entries],
            "FOVFraction": ["1"] * len(self.entries),
        }
        for name, values in per_tilt.items():
            elem = ElementTree.SubElement(root, name)
            elem.text = "\n".join(values)

        ctf = ElementTree.SubElement(root, "CTF")
        params = dict(self.ctf_params)
        params["PixelSize"] = f"{(self.ctf_pixel_size_a or self.pixel_size_a):.9g}"
        for name, value in params.items():
            ElementTree.SubElement(ctf, "Param", Name=name, Value=value)

        n = len(self.entries)
        if self.has_ctf:
            self._grid(root, "GridCTF", [e.defocus_um for e in self.entries])
            self._grid(root, "GridCTFDefocusDelta", [e.defocus_delta_um for e in self.entries])
            self._grid(root, "GridCTFDefocusAngle", [e.defocus_angle_deg for e in self.entries])
            self._grid(root, "GridCTFPhase", [e.phase_shift_pi for e in self.entries])
        else:
            for name in _CTF_GRIDS:
                self._grid(root, name, [0.0])
        for name in _MOVEMENT_GRIDS:
            self._grid(root, name, [0.0])
        for name in _VOLUME_WARP_GRIDS:
            self._grid(root, name, [0.0], four_d=True)
        for name in _ANGLE_GRIDS:
            self._grid(root, name, [0.0])
        for name, default in _OTHER_GRIDS.items():
            self._grid(root, name, [default])
        del n

        ElementTree.indent(root, space="  ")
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ElementTree.tostring(root, encoding="unicode") + "\n"
