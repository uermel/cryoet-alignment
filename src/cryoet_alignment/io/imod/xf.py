from typing import List, Union

import numpy as np
import pandas as pd
import pydantic

from cryoet_alignment.io.base import FileIOBase


class ImodXFInfo(pydantic.BaseModel):
    mxx: float
    mxy: float
    myx: float
    myy: float
    sx: float
    sy: float

    @classmethod
    def from_string(cls, line: str) -> "ImodXFInfo":
        values = line.split()
        mxx = float(values[0])
        mxy = float(values[1])
        myx = float(values[2])
        myy = float(values[3])
        sx = float(values[4])
        sy = float(values[5])
        return cls(mxx=mxx, mxy=mxy, myx=myx, myy=myy, sx=sx, sy=sy)

    def __str__(self) -> str:
        return (
            f"{self.mxx:>12.7f}"
            f"{self.mxy:>12.7f}"
            f"{self.myx:>12.7f}"
            f"{self.myy:>12.7f}"
            f"{self.sx:>12.3f}"
            f"{self.sy:>12.3f}"
        )

    def __iter__(self):
        return iter([self.mxx, self.mxy, self.myx, self.myy, self.sx, self.sy])

    def rot_matrix(self) -> np.ndarray:
        return np.array([[self.mxx, self.mxy], [self.myx, self.myy]])

    def shift(self) -> np.ndarray:
        return np.array([self.sx, self.sy])


class ImodXF(FileIOBase):
    alignments: List[ImodXFInfo]

    @classmethod
    def from_string(cls, text: str) -> "ImodXF":
        lines = text.strip().split("\n")
        alignments = [ImodXFInfo.from_string(line) for line in lines]
        return cls(alignments=alignments)

    def __str__(self):
        return "\n".join(str(alignment) for alignment in self.alignments) + "\n"

    def numpy(self) -> np.ndarray:
        return np.array([list(a) for a in self.alignments])

    def pandas(self) -> pd.DataFrame:
        return pd.DataFrame([a.model_dump() for a in self.alignments])

    def set(self, values: Union[np.ndarray, pd.DataFrame]):
        if isinstance(values, np.ndarray):
            assert values.shape[1] == 6, "Global alignment must have 6 columns."
            self.alignments = [ImodXFInfo(**dict(zip(values.dtype.names, row))) for row in values]
        elif isinstance(values, pd.DataFrame):
            self.alignments = [ImodXFInfo(**row) for _, row in values.iterrows()]
        else:
            raise ValueError("Invalid value type. Must be numpy.ndarray or pandas.DataFrame")
