"""Warp XML tilt-series alignment format (native parser/writer, no dependencies).

Represents only the GLOBAL per-tilt alignment fields (angles, tilt-axis rotation,
2D shifts). Warp's local fields (per-image 2D warp grids and 4D volume warp
grids) are intentionally NOT modeled — they have no analog in the canonical
``Alignment`` model and would be dropped on conversion.

The XML subset handled here mirrors Warp's ``<TiltSeries>`` metadata file:

* root attributes ``ImageDimensionsAngstrom`` / ``VolumeDimensionsAngstrom``
  (comma-separated Å; absent in older Warp exports, in which case they read
  as zeros),
* newline-separated per-tilt text elements ``Angles`` (defines the tilt
  count), ``AxisAngle``, ``AxisOffsetX``, ``AxisOffsetY`` (offsets in Å),
* ``<CTF><Param Name="PixelSize" Value="…"/></CTF>`` as a pixel-size source.

Everything else in a real Warp export (doses, CTF fits, warp grids) is ignored
on read and omitted on write; Warp/warpylib readers default all omitted
elements. Note for downstream tools: Warp conventionally expects the tilt
stack at ``{xml_dir}/tiltstack/{stem}/{stem}.st`` — this module only handles
the XML.
"""

from typing import List, Optional
from xml.etree import ElementTree

from pydantic import BaseModel

from cryoet_alignment.io.base import PATH_TYPE, FileIOBase


class WarpAlignmentEntry(BaseModel):
    """Per-tilt alignment data in Warp convention.

    Attributes:
        z_index: Section index in the FINAL tilt series (0-based).
        tilt_angle: Tilt angle in degrees.
        tilt_axis_angle: Per-tilt tilt-axis rotation angle in degrees.
        tilt_axis_offset_x: X translation in Angstroms.
        tilt_axis_offset_y: Y translation in Angstroms.
    """

    z_index: int
    tilt_angle: float
    tilt_axis_angle: float
    tilt_axis_offset_x: float
    tilt_axis_offset_y: float


def _attr_floats(root: ElementTree.Element, name: str, count: int) -> List[float]:
    """Comma-separated float root attribute; zeros when absent (older exports)."""
    raw = root.get(name)
    if raw is None or not raw.strip():
        return [0.0] * count
    values = [float(v) for v in raw.split(",")]
    if len(values) != count:
        raise ValueError(f"attribute {name} has {len(values)} values, expected {count}")
    return values


def _per_tilt_floats(root: ElementTree.Element, name: str, n_tilts: int) -> List[float]:
    """Newline-separated per-tilt float element; zeros when absent."""
    elem = root.find(name)
    if elem is None or not (elem.text or "").strip():
        return [0.0] * n_tilts
    values = [float(v) for v in elem.text.split() if v.strip()]
    if len(values) != n_tilts:
        raise ValueError(f"element {name} has {len(values)} values, expected {n_tilts} (from Angles)")
    return values


def _ctf_pixel_size(root: ElementTree.Element) -> Optional[float]:
    for param in root.findall("./CTF/Param"):
        if param.get("Name") == "PixelSize" and param.get("Value"):
            return float(param.get("Value"))
    return None


class WarpAlignment(FileIOBase):
    """Warp tilt-series alignment metadata.

    Attributes:
        n_tilts: Number of tilt images.
        pixel_size_a: Pixel size in Angstroms (Å/px).
        image_dimensions_physical: ``[width_a, height_a]`` in Angstroms.
        volume_dimensions_physical: ``[x_a, y_a, z_a]`` in Angstroms (target reconstruction volume).
        entries: Per-tilt alignment entries.
    """

    n_tilts: int
    pixel_size_a: float
    image_dimensions_physical: List[float]
    volume_dimensions_physical: List[float]
    entries: List[WarpAlignmentEntry]

    @classmethod
    def from_string(cls, text: str, pixel_size_a: Optional[float] = None) -> "WarpAlignment":
        """Parse a Warp tilt-series XML.

        Args:
            text: XML content (a leading UTF-8 BOM, as written by Warp, is tolerated).
            pixel_size_a: Tilt-image pixel size in Å/px. The Warp XML stores
                per-tilt shift offsets in Å; ``Alignment.from_warp`` needs to
                divide those by the per-image pixel size to recover pixel
                shifts for ``.aln`` round-trips. When not given, the value is
                read from the XML's ``<CTF><Param Name="PixelSize"/>`` entry —
                correct for real Warp exports, but note that XMLs written by
                third-party tools may carry a placeholder there, so an
                explicit value always wins. If neither is available we refuse
                to guess: a silent 0 propagates into ``Alignment.from_warp``
                as a ``/0 → 0.0`` shift, which silently destroys the alignment.
        """
        root = ElementTree.fromstring(text.lstrip("\ufeff"))

        angles_elem = root.find("Angles")
        if angles_elem is None or not (angles_elem.text or "").strip():
            raise ValueError("no <Angles> element — not a Warp tilt-series XML")
        angles = [float(v) for v in angles_elem.text.split() if v.strip()]
        n_tilts = len(angles)

        image_dims = _attr_floats(root, "ImageDimensionsAngstrom", 2)
        volume_dims = _attr_floats(root, "VolumeDimensionsAngstrom", 3)
        axis_angles = _per_tilt_floats(root, "AxisAngle", n_tilts)
        offsets_x = _per_tilt_floats(root, "AxisOffsetX", n_tilts)
        offsets_y = _per_tilt_floats(root, "AxisOffsetY", n_tilts)

        if pixel_size_a is None:
            pixel_size_a = _ctf_pixel_size(root)
        if pixel_size_a is None:
            raise ValueError(
                "Cannot derive pixel_size_a — the Warp XML carries no "
                "<CTF><Param Name='PixelSize'/> entry. Pass "
                "`pixel_size_a=<source_stack_Å/px>` to WarpAlignment.from_file/"
                "from_string (or use Alignment.from_warp_file with the same parameter).",
            )

        entries = [
            WarpAlignmentEntry(
                z_index=i,
                tilt_angle=angles[i],
                tilt_axis_angle=axis_angles[i],
                tilt_axis_offset_x=offsets_x[i],
                tilt_axis_offset_y=offsets_y[i],
            )
            for i in range(n_tilts)
        ]

        return cls(
            n_tilts=n_tilts,
            pixel_size_a=float(pixel_size_a),
            image_dimensions_physical=image_dims,
            volume_dimensions_physical=volume_dims,
            entries=entries,
        )

    @classmethod
    def from_file(cls, file_path: PATH_TYPE, pixel_size_a: Optional[float] = None) -> "WarpAlignment":
        """Load from a Warp tilt-series XML file (see ``from_string``)."""
        with open(file_path, "r") as file:
            return cls.from_string(file.read(), pixel_size_a=pixel_size_a)

    def __str__(self) -> str:
        """Serialize to a Warp tilt-series XML string.

        Only the modeled subset is written (dimension attributes, the four
        per-tilt elements, and the CTF PixelSize parameter); Warp/warpylib
        readers default everything omitted.
        """
        root = ElementTree.Element("TiltSeries")
        img = self.image_dimensions_physical
        vol = self.volume_dimensions_physical
        root.set("ImageDimensionsAngstrom", f"{img[0]:.9g}, {img[1]:.9g}")
        root.set("VolumeDimensionsAngstrom", f"{vol[0]:.9g}, {vol[1]:.9g}, {vol[2]:.9g}")

        per_tilt = {
            "Angles": [e.tilt_angle for e in self.entries],
            "AxisAngle": [e.tilt_axis_angle for e in self.entries],
            "AxisOffsetX": [e.tilt_axis_offset_x for e in self.entries],
            "AxisOffsetY": [e.tilt_axis_offset_y for e in self.entries],
        }
        for name, values in per_tilt.items():
            elem = ElementTree.SubElement(root, name)
            elem.text = "\n".join(f"{v:.9g}" for v in values)

        ctf = ElementTree.SubElement(root, "CTF")
        ElementTree.SubElement(ctf, "Param", Name="PixelSize", Value=f"{self.pixel_size_a:.9g}")

        ElementTree.indent(root, space="  ")
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ElementTree.tostring(root, encoding="unicode") + "\n"
