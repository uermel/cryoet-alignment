"""Warp ``.tomostar`` tilt-series descriptor (WarpTools ``ts_import`` output).

A STAR file with one unnamed ``data_`` loop, one row per tilt movie
(WarpLib TiltSeries constructor): ``_wrpMovieName`` (movie path relative to
the tomostar directory), ``_wrpAngleTilt`` (degrees), ``_wrpAxisAngle``
(degrees), ``_wrpDose`` (accumulated dose BEFORE the image, e/A^2). Warp
reads ``wrpAngleTilt``, ``wrpDose`` and a non-empty ``wrpMovieName``;
``wrpAxisAngle`` and the ``ts_import`` statistics columns
(``_wrpAverageIntensity``, ``_wrpMaskedFraction``) are optional and preserved.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from cryoet_alignment.io.base import FileIOBase

_KNOWN = {
    "_wrpMovieName": "movie_name",
    "_wrpAngleTilt": "angle_tilt",
    "_wrpAxisAngle": "axis_angle",
    "_wrpDose": "dose",
    "_wrpAverageIntensity": "average_intensity",
    "_wrpMaskedFraction": "masked_fraction",
}


class TomostarRow(BaseModel):
    """One tilt movie of a tomostar.

    Attributes:
        movie_name (str): Movie path relative to the tomostar directory.
        angle_tilt (float): Tilt angle in degrees (Warp convention, as in the tilt-series XML ``Angles``).
        axis_angle (float): Tilt-axis angle in degrees.
        dose (float): Accumulated dose before this image in e/A^2.
        average_intensity (float): ``ts_import`` statistic; None when absent.
        masked_fraction (float): ``ts_import`` statistic; None when absent.
        extra (dict): Unknown columns, label -> text value, preserved verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    movie_name: str
    angle_tilt: float
    axis_angle: float = 0.0
    dose: float = 0.0
    average_intensity: Optional[float] = None
    masked_fraction: Optional[float] = None
    extra: Dict[str, str] = {}


class WarpTomostar(FileIOBase):
    """Warp ``.tomostar`` file.

    Attributes:
        rows (List[TomostarRow]): Tilt movies in file order (Warp keeps this order).
    """

    model_config = ConfigDict(extra="forbid")

    rows: List[TomostarRow]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def movie_names(self) -> List[str]:
        return [r.movie_name for r in self.rows]

    def model_post_init(self, context, /) -> None:
        if not self.rows:
            raise ValueError("tomostar contains no rows")
        for r in self.rows:
            if not r.movie_name:
                raise ValueError("tomostar row with an empty wrpMovieName (Warp requires one)")

    @classmethod
    def from_string(cls, text: str) -> "WarpTomostar":
        labels: List[str] = []
        rows: List[TomostarRow] = []
        in_loop = False
        for raw in text.lstrip("﻿").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                labels, in_loop = [], False
                continue
            if line == "loop_":
                in_loop = True
                continue
            if line.startswith("_"):
                labels.append(line.split()[0])
                continue
            if not in_loop or not labels:
                raise ValueError(f"unexpected tomostar line outside a loop: {raw!r}")
            parts = line.split()
            if len(parts) != len(labels):
                raise ValueError(f"tomostar row has {len(parts)} values for {len(labels)} labels: {raw!r}")
            fields: Dict[str, object] = {}
            extra: Dict[str, str] = {}
            for label, value in zip(labels, parts):
                key = _KNOWN.get(label)
                if key is None:
                    extra[label] = value
                elif key == "movie_name":
                    fields[key] = value
                else:
                    fields[key] = float(value)
            if "movie_name" not in fields or "angle_tilt" not in fields:
                raise ValueError("tomostar needs _wrpMovieName and _wrpAngleTilt columns")
            rows.append(TomostarRow(extra=extra, **fields))
        return cls(rows=rows)

    def __str__(self) -> str:
        labels = ["_wrpMovieName", "_wrpAngleTilt", "_wrpAxisAngle", "_wrpDose"]
        has_stats = all(r.average_intensity is not None and r.masked_fraction is not None for r in self.rows)
        if has_stats:
            labels += ["_wrpAverageIntensity", "_wrpMaskedFraction"]
        extra_labels = list(self.rows[0].extra) if self.rows[0].extra else []
        labels += extra_labels
        out = ["", "data_", "", "loop_"]
        out += [f"{label} #{i + 1}" for i, label in enumerate(labels)]
        for r in self.rows:
            cells = [r.movie_name, f"{r.angle_tilt:.2f}", f"{r.axis_angle:.3f}", f"{r.dose:.6g}"]
            if has_stats:
                cells += [f"{r.average_intensity:.6g}", f"{r.masked_fraction:.3f}"]
            cells += [r.extra.get(label, "") for label in extra_labels]
            out.append("  " + "  ".join(cells))
        return "\n".join(out) + "\n"
