import re
from typing import Optional, Tuple

from cryoet_alignment.io.base import FileIOBase


class ImodNEWSTCOM(FileIOBase):
    AntialiasFilter: int = -1
    InputFile: str
    OutputFile: str
    TransformFile: str
    TaperAtFill: Tuple[int, int] = (0, 0)
    AdjustOrigin: Optional[bool] = True
    OffsetsInXandY: Tuple[float, float] = (0.0, 0.0)
    DistortionField: Optional[str] = None
    ImagesAreBinned: float = 1.0
    BinByFactor: Optional[int] = 1
    GradientFile: Optional[str] = None

    @classmethod
    def from_string(cls, string: str) -> "ImodNEWSTCOM":
        patterns = {
            "AntialiasFilter": (r"^AntialiasFilter\s+(-?\d+)$", (1,)),
            "InputFile": (r"^InputFile\s+(.+)$", (1,)),
            "OutputFile": (r"^OutputFile\s+(.+)$", (1,)),
            "TransformFile": (r"^TransformFile\s+(.+)$", (1,)),
            "TaperAtFill": (r"^TaperAtFill\s+(\d+),(\d+)$", (1, 2)),
            "AdjustOrigin": (r"^AdjustOrigin\s*$", ()),
            "OffsetsInXandY": (r"^OffsetsInXandY\s+(-?[\d.]+),(-?[\d.]+)$", (1, 2)),
            "DistortionField": (r"^DistortionField\s+(.+)$", (1,)),
            "ImagesAreBinned": (r"^ImagesAreBinned\s+([\d.]+)$", (1,)),
            "BinByFactor": (r"^BinByFactor\s+(\d+)$", (1,)),
            "GradientFile": (r"^GradientFile\s+(.+)$", (1,)),
        }

        params = {}

        for key, (pattern, groups) in patterns.items():
            match = re.search(pattern, string, re.M)
            if match:
                if groups:
                    params[key] = match.group(*groups)
                else:
                    params[key] = True
            else:
                params[key] = None

        return cls(**params)

    def __str__(self) -> str:
        outputs = "$setenv IMOD_OUTPUT_FORMAT MRC\n$newstack -StandardInput\n"
        outputs += f"AntialiasFilter\t{self.AntialiasFilter}\n"
        outputs += f"InputFile\t{self.InputFile}\n"
        outputs += f"OutputFile\t{self.OutputFile}\n"
        outputs += f"TransformFile\t{self.TransformFile}\n"
        outputs += f"TaperAtFill\t{self.TaperAtFill[0]},{self.TaperAtFill[1]}\n"
        if self.AdjustOrigin:
            outputs += "AdjustOrigin\n"
        outputs += f"OffsetsInXandY\t{self.OffsetsInXandY[0]},{self.OffsetsInXandY[1]}\n"
        if self.DistortionField:
            outputs += f"DistortionField\t{self.DistortionField}\n"
        else:
            outputs += "#DistortionField\t.idf\n"
        outputs += f"ImagesAreBinned\t{self.ImagesAreBinned}\n"
        outputs += f"BinByFactor\t{self.BinByFactor}\n"
        if self.GradientFile:
            outputs += f"GradientFile\t{self.GradientFile}\n"
        else:
            outputs += "#GradientFile\t.maggrad\n"
        outputs += "$if (-e ./savework) ./savework\n"
        return outputs
