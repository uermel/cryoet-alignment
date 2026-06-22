"""Warp/warpylib XML tilt-series alignment format.

Represents only the GLOBAL per-tilt alignment fields (angles, tilt-axis rotation,
2D shifts). Warp's local fields (per-image 2D warp grids of shape (3,3) and 3D
volume warp grids of shape (3,3,2,10)) are intentionally NOT modeled — they have
no analog in the canonical ``Alignment`` model and would be dropped on conversion.

XML I/O is delegated to ``warpylib.TiltSeries`` (optional dependency: install with
``pip install cryoet-alignment[warp]``) so this module stays a thin schema wrapper
rather than reimplementing Warp's parser.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from cryoet_alignment.io.base import PATH_TYPE, FileIOBase

_WARPYLIB_INSTALL_HINT = (
    "Warp format support requires warpylib. Install with: pip install cryoet-alignment[warp]"
)


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


class WarpAlignment(FileIOBase):
    """Warp/warpylib tilt-series alignment metadata.

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
    def from_file(
        cls,
        file_path: PATH_TYPE,
        pixel_size_a: Optional[float] = None,
    ) -> "WarpAlignment":
        """Load a WarpAlignment from a warpylib XML file.

        Args:
            file_path: Path to the warpylib TiltSeries XML.
            pixel_size_a: Tilt-image pixel size in Å/px. The Warp XML stores
                per-tilt shift offsets in Å; ``Alignment.from_warp`` needs to
                divide those by the per-image pixel size to recover pixel
                shifts for ``.aln`` round-trips. The XML alone doesn't carry
                this number — warpylib derives it from the source stack at
                runtime — so callers that need a faithful round-trip must
                pass it explicitly. If not provided, we attempt to read it
                from the warpylib API and raise with an actionable message
                if that fails (rather than silently returning a 0 that
                zeroes all per-tilt shifts downstream).
        """
        try:
            import torch  # noqa: F401
            from warpylib import TiltSeries
        except ImportError as e:
            raise ImportError(_WARPYLIB_INSTALL_HINT) from e

        ts = TiltSeries(path=str(file_path))

        image_dims = [float(v) for v in ts.image_dimensions_physical]
        volume_dims = [float(v) for v in ts.volume_dimensions_physical]

        if pixel_size_a is None:
            # warpylib's `load_image_dimensions(original_pixel_size)` (a) requires
            # an argument we don't have here, and (b) returns None even when
            # called correctly — it just side-effects attributes on the
            # TiltSeries. So we can't derive pixel size from the XML alone.
            # Refuse to guess: a silent 0 propagates into Alignment.from_warp
            # as a `/0 → 0.0` shift, which silently destroys the alignment.
            raise ValueError(
                f"Cannot derive pixel_size_a from {file_path} — the Warp XML "
                "doesn't store it directly and warpylib's load_image_dimensions "
                "API requires an explicit `original_pixel_size`. Pass "
                "`pixel_size_a=<source_stack_Å/px>` to WarpAlignment.from_file "
                "(or use Alignment.from_warp_file with the same parameter).",
            )
        pixel_size_a = float(pixel_size_a)

        n_tilts = int(ts.n_tilts)
        entries = []
        for i in range(n_tilts):
            entries.append(
                WarpAlignmentEntry(
                    z_index=i,
                    tilt_angle=float(ts.angles[i]),
                    tilt_axis_angle=float(ts.tilt_axis_angles[i]),
                    tilt_axis_offset_x=float(ts.tilt_axis_offset_x[i]),
                    tilt_axis_offset_y=float(ts.tilt_axis_offset_y[i]),
                ),
            )

        return cls(
            n_tilts=n_tilts,
            pixel_size_a=pixel_size_a,
            image_dimensions_physical=image_dims,
            volume_dimensions_physical=volume_dims,
            entries=entries,
        )

    def to_file(self, file_path: PATH_TYPE) -> None:
        """Write a WarpAlignment to a warpylib XML file.

        Note: this also requires the corresponding tilt stack to live at the
        warpylib-conventional location ``{file_path.parent}/tiltstack/{stem}/{stem}.st``
        for downstream tools that load the stack. This method only writes the XML;
        the caller is responsible for the stack file/symlink.
        """
        try:
            import torch
            from warpylib import TiltSeries
        except ImportError as e:
            raise ImportError(_WARPYLIB_INSTALL_HINT) from e

        xml_path = Path(file_path)
        ts = TiltSeries(path=str(xml_path), n_tilts=self.n_tilts)

        ts.angles = torch.tensor([e.tilt_angle for e in self.entries], dtype=torch.float32)
        ts.tilt_axis_angles = torch.tensor(
            [e.tilt_axis_angle for e in self.entries], dtype=torch.float32,
        )
        ts.tilt_axis_offset_x = torch.tensor(
            [e.tilt_axis_offset_x for e in self.entries], dtype=torch.float32,
        )
        ts.tilt_axis_offset_y = torch.tensor(
            [e.tilt_axis_offset_y for e in self.entries], dtype=torch.float32,
        )

        ts.image_dimensions_physical = torch.tensor(
            self.image_dimensions_physical, dtype=torch.float32,
        )
        ts.volume_dimensions_physical = torch.tensor(
            self.volume_dimensions_physical, dtype=torch.float32,
        )

        ts.save_meta(str(xml_path))

    @classmethod
    def from_string(cls, text: str) -> "WarpAlignment":
        """Parse from a string of XML content. Less common than ``from_file``;
        warpylib's TiltSeries is path-based, so this writes to a temp file."""
        try:
            from warpylib import TiltSeries  # noqa: F401
        except ImportError as e:
            raise ImportError(_WARPYLIB_INSTALL_HINT) from e

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(text)
            tmp_path = f.name
        try:
            return cls.from_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def __str__(self) -> str:
        """Serialize to an XML string by writing through a temp file."""
        try:
            from warpylib import TiltSeries  # noqa: F401
        except ImportError as e:
            raise ImportError(_WARPYLIB_INSTALL_HINT) from e

        import tempfile

        with tempfile.NamedTemporaryFile(mode="r", suffix=".xml", delete=False) as f:
            tmp_path = f.name
        try:
            self.to_file(tmp_path)
            with open(tmp_path) as f:
                return f.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
