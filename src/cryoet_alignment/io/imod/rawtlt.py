from typing import List

import numpy as np
import pandas as pd

from cryoet_alignment.io.base import FileIOBase


class ImodRAWTLT(FileIOBase):
    angles: List[float]

    @classmethod
    def from_string(cls, text: str) -> "ImodRAWTLT":
        lines = text.strip().split("\n")
        angles = [float(line) for line in lines]
        return cls(angles=angles)

    def __str__(self) -> str:
        return "\n".join(str(angle) for angle in self.angles) + "\n"

    def numpy(self) -> np.ndarray:
        return np.array(self.angles)

    def pandas(self):
        return pd.DataFrame(self.angles, columns=["angles"])


class ImodTLT(ImodRAWTLT):
    pass


class ImodXTILT(ImodRAWTLT):
    pass
