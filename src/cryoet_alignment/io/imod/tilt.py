import re
from typing import List, Optional, Tuple

from pydantic import Field

from cryoet_alignment.io.base import FileIOBase


def imod_range_to_list(text: str) -> List[int]:
    """Convert a range of numbers in IMOD format to a list of integers.

    e.g. 1,2-6,8,10-12 -> [1, 2, 3, 4, 5, 6, 8, 10, 11, 12]

    Args:
        text (str): The text to convert.

    Returns:
        List[int]: The list of integers.
    """
    if text == "" or text is None:
        return []

    parts = text.split(",")
    out = []

    for p in parts:
        if "-" in p:
            start, end = p.split("-")
            out += list(range(int(start), int(end) + 1))
        else:
            out.append(int(p))

    return out


class ImodTILTCOM(FileIOBase):
    """Class to represent an IMOD tilt command file.

    Attributes:
        InputProjections (str): The input projections file.
        OutputFile (str): The output tomogram file.
        IMAGEBINNED (int): Whether the image was binned.
        TILTFILE (str): The tilt file (`{basename}.tlt`)
        THICKNESS (int): The thickness of the tomogram in pixels of the unbinned input tilt series.
        RADIAL (Tuple[float, float]): The radial filter cutoff and falloff.
        FalloffIsTrueSigma (int): Whether the falloff is true sigma.
        XAXISTILT (float): The global additional x-axis tilt.
        SCALE (Tuple[float, float]): TBD
        PERPENDICULAR (bool): TBD
        MODE (int): The output data type (MRC modes).
        FULLIMAGE (Tuple[int, int]): The full image size (input tilt series).
        SUBSETSTART (Tuple[int, int]): TBD
        AdjustOrigin (bool): TBD
        ActionIfGPUFails (Tuple[int, int]): TBD
        XTILTFILE (str): Per section x-axis tilt file.
        OFFSET (float): TBD
        SHIFT (Tuple[float, float]): TBD
        EXCLUDELIST2 (Optional[List[int]]): The list of sections to exclude (1-based index into tilt series).

    """

    InputProjections: str
    OutputFile: str
    IMAGEBINNED: int = 1
    TILTFILE: str
    THICKNESS: int
    RADIAL: Tuple[float, float] = (0.35, 0.035)
    FalloffIsTrueSigma: int = 1
    XAXISTILT: float = 0.0
    SCALE: Tuple[float, float] = (0.0, 0.1)
    PERPENDICULAR: bool = True
    MODE: int = Field(alias="Mode", default=2)  # int = 2
    FULLIMAGE: Tuple[int, int]
    SUBSETSTART: Tuple[int, int] = (0, 0)
    AdjustOrigin: bool = True
    ActionIfGPUFails: Optional[Tuple[int, int]] = (1, 2)
    XTILTFILE: str
    OFFSET: Optional[float] = 0.0
    SHIFT: Optional[Tuple[float, float]] = (0.0, 0.0)
    EXCLUDELIST2: Optional[List[int]] = None

    @classmethod
    def from_string(cls, text: str):
        patterns = {
            "InputProjections": (r"^InputProjections\s+(.+)$", (1,)),
            "OutputFile": (r"^OutputFile\s+(.+)$", (1,)),
            "IMAGEBINNED": (r"^IMAGEBINNED\s+(\d+)$", (1,)),
            "TILTFILE": (r"^TILTFILE\s+(.+)$", (1,)),
            "THICKNESS": (r"^THICKNESS\s+([\d.]+)$", (1,)),
            "RADIAL": (r"^RADIAL\s+([\d.]+)\s+([\d.]+)$", (1, 2)),
            "FalloffIsTrueSigma": (r"^FalloffIsTrueSigma\s+(\d+)$", (1,)),
            "XAXISTILT": (r"^XAXISTILT\s+([\d.]+)$", (1,)),
            "SCALE": (r"^SCALE\s+([\d.]+)\s+([\d.]+)$", (1, 2)),
            "PERPENDICULAR": (r"^PERPENDICULAR.*$", ()),
            "MODE": (r"^MODE\s+(\d+)$", (1,)),
            "FULLIMAGE": (r"^FULLIMAGE\s+(\d+)\s+(\d+)$", (1, 2)),
            "SUBSETSTART": (r"^SUBSETSTART\s+([-\d]+)\s+([-\d]+)$", (1, 2)),
            "AdjustOrigin": (r"^AdjustOrigin.*$", ()),
            "ActionIfGPUFails": (r"^ActionIfGPUFails\s+(\d+),(\d+)$", (1, 2)),
            "XTILTFILE": (r"^XTILTFILE\s+(.+)$", (1,)),
            "OFFSET": (r"^OFFSET\s+(-?[\d.]+)$", (1,)),
            "SHIFT": (r"^SHIFT\s+(-?[\d.]+)\s+(-?[\d.]+)$", (1, 2)),
            "EXCLUDELIST2": (r"^(EXCLUDELIST2|EXCLUDELIST|EXCLUDE)\s+(.+)$", (2,)),
        }

        params = {}
        for key, (pattern, groups) in patterns.items():
            match = re.search(pattern, text, re.M)
            if match:
                if groups:
                    if key == "EXCLUDELIST2":
                        params[key] = imod_range_to_list(match.group(*groups))
                    else:
                        params[key] = match.group(*groups)
                else:
                    params[key] = True
            else:
                params[key] = None

        return cls(**params)

    def __str__(self):
        outputs = "$setenv IMOD_OUTPUT_FORMAT MRC\n$tilt -StandardInput\n"
        outputs += f"InputProjections {self.InputProjections}\n"
        outputs += f"OutputFile {self.OutputFile}\n"
        outputs += f"IMAGEBINNED {self.IMAGEBINNED}\n"
        outputs += f"TILTFILE {self.TILTFILE}\n"
        outputs += f"THICKNESS {self.THICKNESS}\n"
        outputs += f"RADIAL {self.RADIAL[0]:.2f} {self.RADIAL[1]:.3f}\n"
        outputs += f"FalloffIsTrueSigma {self.FalloffIsTrueSigma}\n"
        outputs += f"XAXISTILT {self.XAXISTILT}\n"
        outputs += f"SCALE {self.SCALE[0]:.1f} {self.SCALE[1]:.1f}\n"
        if self.PERPENDICULAR:
            outputs += "PERPENDICULAR\n"
        outputs += f"MODE {self.MODE}\n"
        outputs += f"FULLIMAGE {self.FULLIMAGE[0]} {self.FULLIMAGE[1]}\n"
        outputs += f"SUBSETSTART {self.SUBSETSTART[0]} {self.SUBSETSTART[1]}\n"
        if self.AdjustOrigin:
            outputs += "AdjustOrigin\n"
        outputs += f"ActionIfGPUFails {self.ActionIfGPUFails[0]},{self.ActionIfGPUFails[1]}\n"
        outputs += f"XTILTFILE {self.XTILTFILE}\n"
        outputs += f"OFFSET {self.OFFSET}\n"
        outputs += f"SHIFT {self.SHIFT[0]} {self.SHIFT[1]}\n"
        if self.EXCLUDELIST2:
            outputs += f"EXCLUDELIST2 {','.join(map(str, self.EXCLUDELIST2))}\n"
        outputs += "$if (-e ./savework) ./savework\n"
        return outputs
