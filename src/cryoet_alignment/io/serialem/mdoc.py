"""SerialEM ``.mdoc`` autodoc files written next to a tilt-series stack.

An mdoc is a list of ``key = value`` lines. Lines before the first section form the header (``PixelSpacing``,
``Voltage``, ``ImageFile``, ``ImageSize``, ``DataMode``); ``[T = ...]`` lines are free-text titles, one of which
usually carries ``Tilt axis angle = ...``; each ``[ZValue = n]`` opens the section for image ``n`` of the stack
(0-based). SerialEM writes the stack, and so the ``ZValue`` order, in acquisition order.

Per-section keys used here (SerialEM autodoc documentation):

* ``TiltAngle``: stage tilt angle in degrees.
* ``ExposureDose``: dose of this image in e/A^2 (0 when SerialEM's dose is not calibrated).
* ``PriorRecordDose``: dose of the Record images acquired before this one in the same tilt series, in e/A^2
  (exclusive of this image; SerialEM 4.0 and later).
* ``DateTime``: acquisition time, ``dd-Mon-yy  HH:MM:SS``.
* ``SubFramePath``: the movie of this image, as a path on the acquisition computer (often a Windows path).
* ``PixelSpacing``, ``Defocus``, ``TargetDefocus`` (micrometres, negative is underfocus), ``ExposureTime`` (s),
  ``NumSubFrames``, ``RotationAngle``, ``Magnification``.

Every other key is kept verbatim in ``extra`` and written back unchanged. Montage files (``[MontSection = n]``) are
not tilt series and are refused.
"""

import datetime
import math
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from cryoet_alignment.io.base import FileIOBase

_SECTION_RE = re.compile(r"^\[\s*(\w+)\s*=\s*(.*?)\s*\]$")
_TILT_AXIS_RE = re.compile(r"tilt\s*axis\s*angle\s*=\s*([-+0-9.eE]+)", re.IGNORECASE)
_DATETIME_FORMATS = ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S")

#: Per-section keys read into typed fields: mdoc key -> (field, type).
SECTION_KEYS = {
    "TiltAngle": ("tilt_angle", float),
    "ExposureDose": ("exposure_dose", float),
    "PriorRecordDose": ("prior_record_dose", float),
    "PixelSpacing": ("pixel_spacing", float),
    "Defocus": ("defocus", float),
    "TargetDefocus": ("target_defocus", float),
    "ExposureTime": ("exposure_time", float),
    "RotationAngle": ("rotation_angle", float),
    "Magnification": ("magnification", int),
    "NumSubFrames": ("num_sub_frames", int),
    "SubFramePath": ("sub_frame_path", str),
    "DateTime": ("date_time", str),
}

#: Header keys read into typed fields: mdoc key -> (field, type).
HEADER_KEYS = {
    "PixelSpacing": ("pixel_spacing", float),
    "Voltage": ("voltage", float),
    "ImageFile": ("image_file", str),
    "ImageSize": ("image_size", "int2"),
    "DataMode": ("data_mode", int),
}


def _parse(value: str, kind):
    if kind == "int2":
        x, y = value.split()[:2]
        return int(x), int(y)
    if kind is int:
        return int(float(value))
    if kind is float:
        return float(value.split()[0])
    return value


def _format(value) -> str:
    if isinstance(value, tuple):
        return " ".join(str(v) for v in value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    return str(value)


def _split_line(line: str) -> Tuple[str, str]:
    key, _, value = line.partition("=")
    return key.strip(), value.strip()


class MdocSection(BaseModel):
    """One ``[ZValue = n]`` section: one image of the stack.

    Attributes:
        z_value: 0-based index of the image in the stack.
        tilt_angle: Stage tilt angle in degrees.
        exposure_dose: Dose of this image in e/A^2.
        prior_record_dose: Dose of the earlier Record images of the tilt series in e/A^2.
        pixel_spacing: Pixel size in Angstrom.
        defocus: Defocus in micrometres as SerialEM reports it (negative is underfocus).
        target_defocus: Target defocus in micrometres.
        exposure_time: Exposure time in seconds.
        rotation_angle: Image rotation angle in degrees.
        magnification: Nominal magnification.
        num_sub_frames: Number of movie frames.
        sub_frame_path: Path of the movie, as written on the acquisition computer.
        date_time: Acquisition time as written (``dd-Mon-yy  HH:MM:SS``).
        extra: All other keys, verbatim and in file order.
    """

    model_config = ConfigDict(extra="forbid")

    z_value: int
    tilt_angle: Optional[float] = None
    exposure_dose: Optional[float] = None
    prior_record_dose: Optional[float] = None
    pixel_spacing: Optional[float] = None
    defocus: Optional[float] = None
    target_defocus: Optional[float] = None
    exposure_time: Optional[float] = None
    rotation_angle: Optional[float] = None
    magnification: Optional[int] = None
    num_sub_frames: Optional[int] = None
    sub_frame_path: Optional[str] = None
    date_time: Optional[str] = None
    extra: Dict[str, str] = {}

    @property
    def acquired_at(self) -> Optional[datetime.datetime]:
        """``date_time`` parsed, or None when absent or unparseable."""
        if not self.date_time:
            return None
        text = " ".join(self.date_time.split())
        for fmt in _DATETIME_FORMATS:
            try:
                return datetime.datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @property
    def sub_frame_name(self) -> Optional[str]:
        """File name of the movie, with the acquisition computer's directories (``/`` or ``\\``) stripped."""
        if not self.sub_frame_path:
            return None
        return re.split(r"[\\/]", self.sub_frame_path)[-1]


class SerialEMMdoc(FileIOBase):
    """A SerialEM tilt-series ``.mdoc`` file.

    Attributes:
        pixel_spacing: Header ``PixelSpacing`` in Angstrom.
        voltage: Header ``Voltage`` in kV.
        image_file: Header ``ImageFile``, the stack the mdoc describes.
        image_size: Header ``ImageSize`` (width, height) in px.
        data_mode: Header ``DataMode`` (MRC mode).
        header_extra: Other header keys, verbatim.
        titles: ``[T = ...]`` title lines, without the brackets.
        sections: The ``[ZValue = n]`` sections in file order.
    """

    model_config = ConfigDict(extra="forbid")

    pixel_spacing: Optional[float] = None
    voltage: Optional[float] = None
    image_file: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    data_mode: Optional[int] = None
    header_extra: Dict[str, str] = {}
    titles: List[str] = []
    sections: List[MdocSection]

    def model_post_init(self, context, /) -> None:
        if not self.sections:
            raise ValueError("mdoc contains no [ZValue = n] sections")
        z = [s.z_value for s in self.sections]
        if len(set(z)) != len(z):
            raise ValueError(f"mdoc lists a ZValue more than once: {z}")
        for s in self.sections:
            if s.tilt_angle is None or not math.isfinite(s.tilt_angle):
                raise ValueError(f"mdoc section ZValue = {s.z_value} has no finite TiltAngle")

    @property
    def n_sections(self) -> int:
        return len(self.sections)

    @property
    def tilt_angles(self) -> List[float]:
        """Stage tilt angles in section order."""
        return [s.tilt_angle for s in self.sections]

    @property
    def tilt_axis_angle(self) -> Optional[float]:
        """The ``Tilt axis angle`` of the first title that names one, in degrees; None when no title does."""
        for title in self.titles:
            match = _TILT_AXIS_RE.search(title)
            if match:
                return float(match.group(1))
        return None

    def acquisition_order(self) -> List[int]:
        """0-based acquisition order of each section, in section order.

        From ``DateTime`` when every section has a parseable one (ties keep the ``ZValue`` order), otherwise the
        ``ZValue`` order, which is the order SerialEM wrote the images in.
        """
        times = [s.acquired_at for s in self.sections]
        if all(t is not None for t in times):
            keys = [(t, s.z_value) for t, s in zip(times, self.sections)]
        else:
            keys = [(s.z_value,) for s in self.sections]
        ranked = sorted(range(len(keys)), key=lambda i: keys[i])
        order = [0] * len(keys)
        for rank, i in enumerate(ranked):
            order[i] = rank
        return order

    def pre_exposure(self, dose_per_tilt: Optional[float] = None) -> Optional[List[float]]:
        """Dose each image received before it was acquired (exclusive), in e/A^2, in section order.

        Args:
            dose_per_tilt: A uniform dose per image overriding the file's values.

        Returns:
            ``PriorRecordDose`` when every section has one and any is positive; otherwise the running sum of
            ``ExposureDose`` over the earlier images in acquisition order when every section has one and any is
            positive; otherwise None (the file carries no usable dose).
        """
        order = self.acquisition_order()
        if dose_per_tilt is not None:
            return [rank * float(dose_per_tilt) for rank in order]
        prior = [s.prior_record_dose for s in self.sections]
        if all(p is not None for p in prior) and any(p > 0 for p in prior):
            return [float(p) for p in prior]
        dose = [s.exposure_dose for s in self.sections]
        if all(d is not None for d in dose) and any(d > 0 for d in dose):
            by_rank = sorted(range(len(dose)), key=lambda i: order[i])
            pre = [0.0] * len(dose)
            total = 0.0
            for i in by_rank:
                pre[i] = total
                total += float(dose[i])
            return pre
        return None

    @classmethod
    def from_string(cls, text: str) -> "SerialEMMdoc":
        header: Dict[str, object] = {}
        header_extra: Dict[str, str] = {}
        titles: List[str] = []
        sections: List[MdocSection] = []
        current: Optional[Dict[str, object]] = None

        def close():
            if current is not None:
                sections.append(MdocSection(**current))

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            match = _SECTION_RE.match(line)
            if match:
                kind, value = match.group(1), match.group(2)
                if kind == "T":
                    titles.append(value)
                    continue
                if kind == "ZValue":
                    close()
                    current = {"z_value": int(value), "extra": {}}
                    continue
                if kind == "MontSection":
                    raise ValueError("montage mdoc ([MontSection = n]) — not a tilt series")
                raise ValueError(f"unsupported mdoc section [{kind} = {value}]")
            key, value = _split_line(line)
            if current is None:
                if key in HEADER_KEYS:
                    field, kind = HEADER_KEYS[key]
                    header[field] = _parse(value, kind)
                else:
                    header_extra[key] = value
            elif key in SECTION_KEYS:
                field, kind = SECTION_KEYS[key]
                current[field] = _parse(value, kind) if value else None
            else:
                current["extra"][key] = value
        close()
        return cls(**header, header_extra=header_extra, titles=titles, sections=sections)

    def __str__(self) -> str:
        out = []
        for key, (field, _) in HEADER_KEYS.items():
            value = getattr(self, field)
            if value is not None:
                out.append(f"{key} = {_format(value)}")
        out.extend(f"{key} = {value}" for key, value in self.header_extra.items())
        for title in self.titles:
            out.extend(["", f"[T = {title}]"])
        for section in self.sections:
            out.extend(["", f"[ZValue = {section.z_value}]"])
            for key, (field, _) in SECTION_KEYS.items():
                value = getattr(section, field)
                if value is not None:
                    out.append(f"{key} = {_format(value)}")
            out.extend(f"{key} = {value}" for key, value in section.extra.items())
        return "\n".join(out) + "\n\n"
