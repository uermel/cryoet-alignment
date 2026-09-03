"""Warp ``.settings`` file (``WarpTools create_settings``; ``OptionsWarp``).

An XML document ``<Settings>`` with top-level ``<Param Name=".." Value=".."/>``
entries and sections (``Import``, ``CTF``, ``Movement``, ``Grids``, ``Tomo``,
``Picking``, ``Export``, ``Tasks``, ``Filter``) of ``Param`` entries. Warp's
loader falls back to defaults for every missing parameter, and
``ts_reconstruct`` reads ``Import/PixelSize``, ``Import/DataFolder``,
``Import/ProcessingFolder``, ``Import/Extension`` and ``Tomo/DimensionsX/Y/Z``.
``create`` fills the vendored default document (a genuine Warp 2.0
tilt-series settings file) with exactly the parameters ``create_settings``
sets: data/processing folders, extension, pixel size, bin times, gain
options, ``DosePerAngstromFrame`` (= MINUS the per-tilt exposure),
``EERGroupFrames`` (= minus the group count) and the tomogram box.
"""

from importlib import resources
from typing import Dict, List, Optional
from xml.etree import ElementTree

from pydantic import ConfigDict

from cryoet_alignment.io.base import FileIOBase

_DECLARATION = '<?xml version="1.0" encoding="utf-8"?>'


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class WarpSettings(FileIOBase):
    """Warp ``.settings`` document.

    Attributes:
        params (Dict[str, str]): Top-level ``Param`` entries (ProcessCTF, ...), in file order.
        sections (Dict[str, Dict[str, str]]): Section name -> {param name: value}, in file order.
    """

    model_config = ConfigDict(extra="forbid")

    params: Dict[str, str]
    sections: Dict[str, Dict[str, str]]

    # --- access -----------------------------------------------------------------

    def get(self, section: Optional[str], name: str) -> Optional[str]:
        table = self.params if section is None else self.sections.get(section, {})
        return table.get(name)

    def set(self, section: Optional[str], name: str, value) -> None:
        if section is None:
            self.params[name] = _fmt(value)
        else:
            self.sections.setdefault(section, {})[name] = _fmt(value)

    def _float(self, section: str, name: str) -> Optional[float]:
        v = self.get(section, name)
        return float(v) if v not in (None, "") else None

    @property
    def pixel_size_a(self) -> Optional[float]:
        return self._float("Import", "PixelSize")

    @property
    def data_folder(self) -> Optional[str]:
        return self.get("Import", "DataFolder")

    @property
    def processing_folder(self) -> Optional[str]:
        return self.get("Import", "ProcessingFolder")

    @property
    def extension(self) -> Optional[str]:
        return self.get("Import", "Extension")

    @property
    def exposure_per_tilt(self) -> Optional[float]:
        """Per-tilt exposure in e/A^2 (Warp stores its negative in DosePerAngstromFrame)."""
        v = self._float("Import", "DosePerAngstromFrame")
        return -v if v is not None else None

    @property
    def tomo_dims_px(self) -> Optional[List[int]]:
        dims = [self._float("Tomo", f"Dimensions{ax}") for ax in ("X", "Y", "Z")]
        if any(d is None for d in dims):
            return None
        return [int(round(d)) for d in dims]

    @property
    def voltage_kv(self) -> Optional[float]:
        return self._float("CTF", "Voltage")

    @property
    def cs_mm(self) -> Optional[float]:
        return self._float("CTF", "Cs")

    @property
    def amplitude_contrast(self) -> Optional[float]:
        return self._float("CTF", "Amplitude")

    # --- construction ------------------------------------------------------------

    @classmethod
    def default(cls) -> "WarpSettings":
        """The vendored default tilt-series settings document."""
        text = resources.files("cryoet_alignment.io.warp").joinpath("data/warp_tiltseries.settings").read_text()
        return cls.from_string(text)

    @classmethod
    def create(
        cls,
        *,
        pixel_size_a: float,
        exposure_per_tilt: float,
        tomo_dims_px,
        data_folder: str = "tomostar",
        processing_folder: str = "warp_tiltseries",
        extension: str = "*.tomostar",
        bin_times: float = 0.0,
        eer_group_frames: int = 40,
        gain_path: str = "",
        correct_gain: Optional[bool] = None,
        voltage_kv: Optional[float] = None,
        cs_mm: Optional[float] = None,
        amplitude_contrast: Optional[float] = None,
    ) -> "WarpSettings":
        """What ``WarpTools create_settings --folder_data DATA --folder_processing PROC
        --extension EXT --angpix PIX --exposure DOSE --tomo_dimensions XxYxZ`` writes,
        plus optional CTF constants."""
        s = cls.default()
        s.set("Import", "DataFolder", data_folder)
        s.set("Import", "ProcessingFolder", processing_folder)
        s.set("Import", "Extension", extension)
        s.set("Import", "PixelSize", float(pixel_size_a))
        s.set("Import", "BinTimes", float(bin_times))
        s.set("Import", "DosePerAngstromFrame", -float(exposure_per_tilt))
        s.set("Import", "EERGroupFrames", -int(eer_group_frames))
        s.set("Import", "GainPath", gain_path)
        s.set("Import", "CorrectGain", bool(gain_path) if correct_gain is None else correct_gain)
        dims = [int(v) for v in tomo_dims_px]
        if len(dims) != 3:
            raise ValueError("tomo_dims_px must be (X, Y, Z)")
        for ax, v in zip(("X", "Y", "Z"), dims):
            s.set("Tomo", f"Dimensions{ax}", v)
        if voltage_kv is not None:
            s.set("CTF", "Voltage", float(voltage_kv))
        if cs_mm is not None:
            s.set("CTF", "Cs", float(cs_mm))
        if amplitude_contrast is not None:
            s.set("CTF", "Amplitude", float(amplitude_contrast))
        return s

    # --- serialization -----------------------------------------------------------

    @classmethod
    def from_string(cls, text: str) -> "WarpSettings":
        root = ElementTree.fromstring(text.lstrip("﻿"))
        if root.tag != "Settings":
            raise ValueError(f"root element is <{root.tag}>, expected <Settings>")
        params: Dict[str, str] = {}
        sections: Dict[str, Dict[str, str]] = {}
        for child in root:
            if child.tag == "Param":
                params[child.get("Name")] = child.get("Value", "")
            else:
                sections[child.tag] = {p.get("Name"): p.get("Value", "") for p in child.findall("Param")}
        return cls(params=params, sections=sections)

    def __str__(self) -> str:
        out = [_DECLARATION, "<Settings>"]
        for name, value in self.params.items():
            out.append(f'\t<Param Name="{name}" Value="{_escape(value)}" />')
        for section, table in self.sections.items():
            out.append(f"\t<{section}>")
            for name, value in table.items():
                out.append(f'\t\t<Param Name="{name}" Value="{_escape(value)}" />')
            out.append(f"\t</{section}>")
        out.append("</Settings>")
        return "\n".join(out)


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
