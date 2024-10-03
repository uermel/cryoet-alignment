from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from cryoet_alignment.io.base import FileIOBase


class GlobalAlignmentInfo(BaseModel):
    """Global alignment information for one section of a tilt series.

    Attributes:
        sec (int): Section index in the FINAL tilt series used for reconstruction, after removal of DarkFrames (0-based).
        rot (float): Tilt Axis Rotation angle in degrees.
        gmag (float): Magnification change.
        tx (float): X translation in pixels.
        ty (float): Y translation in pixels.
        smean (float): TBD
        sfit (float): TBD
        scale (float): TBD
        base (float): TBD
        tilt (float): Tilt angle in degrees.
    """

    sec: int
    rot: float
    gmag: float = 1.0
    tx: float
    ty: float
    smean: float = 1.0
    sfit: float = 1.0
    scale: float = 1.0
    base: float = 0.0
    tilt: float

    @classmethod
    def from_string(cls, line: str):
        values = line.split()
        sec = int(values[0])
        rot = float(values[1])
        gmag = float(values[2])
        tx = float(values[3])
        ty = float(values[4])
        smean = float(values[5])
        sfit = float(values[6])
        scale = float(values[7])
        base = float(values[8])
        tilt = float(values[9])
        return cls(sec=sec, rot=rot, gmag=gmag, tx=tx, ty=ty, smean=smean, sfit=sfit, scale=scale, base=base, tilt=tilt)

    def __iter__(self):
        return iter(
            [self.sec, self.rot, self.gmag, self.tx, self.ty, self.smean, self.sfit, self.scale, self.base, self.tilt],
        )

    def __str__(self):
        return (
            f"{self.sec:>5}"
            f"{self.rot:>11.4f}"
            f"{self.gmag:>11.5f}"
            f"{self.tx:>11.3f}"
            f"{self.ty:>11.3f}"
            f"{self.smean:>9.2f}"
            f"{self.sfit:>9.2f}"
            f"{self.scale:>9.2f}"
            f"{self.base:>9.2f}"
            f"{self.tilt:>10.2f}"
        )


class DarkFrameInfo(BaseModel):
    """Dark frame information for one section of a tilt series.

    Attributes:
        section_idx (int): Section index in the INPUT tilt series, before removal of DarkFrames (0-based).
        val2 (int): TBD
        angle (float): Tilt angle in degrees.
    """

    section_idx: int
    val2: int
    angle: float

    @classmethod
    def from_string(cls, line: str):
        parts = line.split("=")
        values = parts[1].split()
        section_idx = int(values[0])
        val2 = int(values[1])
        angle = float(values[2])
        return cls(section_idx=section_idx, val2=val2, angle=angle)

    def __iter__(self):
        return iter([self.section_idx, self.val2, self.angle])

    def __str__(self):
        return f"# DarkFrame ={self.section_idx:>6}{self.val2:>5}{self.angle:>9.2f}"


class LocalAlignmentInfo(BaseModel):
    """Local alignment information for a patch on a section of a tilt series.

    Attributes:
        sec_idx (int): Section index in the FINAL tilt series used for reconstruction, after removal of DarkFrames
            (0-based).
        patch_idx (int): Patch index in the section (0-based).
        center_x (float): projected x coordinate of patch-subvolume wrp to section cetner (expected)
        center_y (float): projected y coordinate of patch-subvolume wrp to section cetner (expected)
        shift_x (float): x shift from expected patch center
        shift_y (float): y shift from expected patch center
        is_reliable (float): reliable/unrelaible flag
    """

    sec_idx: int
    patch_idx: int
    center_x: float
    center_y: float
    shift_x: float
    shift_y: float
    is_reliable: float

    @classmethod
    def from_string(cls, line: str):
        values = line.split()
        sec_idx = int(values[0])
        patch_idx = int(values[1])
        center_x = float(values[2])
        center_y = float(values[3])
        shift_x = float(values[4])
        shift_y = float(values[5])
        is_reliable = float(values[6])
        return cls(
            sec_idx=sec_idx,
            patch_idx=patch_idx,
            center_x=center_x,
            center_y=center_y,
            shift_x=shift_x,
            shift_y=shift_y,
            is_reliable=is_reliable,
        )

    def __iter__(self):
        return iter(
            [
                self.sec_idx,
                self.patch_idx,
                self.center_x,
                self.center_y,
                self.shift_x,
                self.shift_y,
                self.is_reliable,
            ],
        )

    def __str__(self):
        return (
            f"{self.sec_idx:>4}"
            f"{self.patch_idx:>4}"
            f"{self.center_x:>9.2f}"
            f"{self.center_y:>10.2f}"
            f"{self.shift_x:>10.2f}"
            f"{self.shift_y:>10.2f}"
            f"{self.is_reliable:>6.1f}"
        )


class AreTomo3ALN(FileIOBase):
    """AreTomo3's alignment file format (.aln). This file contains the global and local alignment information for a
    tilt series.

    Some rules (not enforced in the class yet):
    - The header must be "# AreTomo Alignment"
    - RawSize[2] == len(DarkFrames) + len(GlobalAlignments)
    - len(LocalAlignments) == len(GlobalAlignments) * NumPatches

    Attributes:
        header (str): Header of the file. Should be "# AreTomo Alignment".
        RawSize (Tuple[int, int, int]): Size of the tilt series in pixels (x, y) and sections (z).
        NumPatches (int): Number of patches for local alignment.
        DarkFrames (List[DarkFrameInfo]): List of dark frames (discarded for reconstruction).
        AlphaOffset (float): Alpha offset for the reconstruction.
        BetaOffset (float): Beta offset for the reconstruction.
        GlobalAlignments (List[GlobalAlignmentInfo]): List of global alignments.
        LocalAlignments (List[LocalAlignmentInfo]): List of local alignments.
    """

    header: Optional[str] = "# AreTomo Alignment / Priims bprmMn"
    RawSize: Tuple[int, int, int]
    NumPatches: int
    DarkFrames: List[DarkFrameInfo]
    AlphaOffset: float
    BetaOffset: float
    GlobalAlignments: List[GlobalAlignmentInfo]
    LocalAlignments: Optional[List[LocalAlignmentInfo]] = None

    @classmethod
    def from_string(cls, text: str) -> "AreTomo3ALN":
        text = text.strip()
        lines = text.splitlines()

        header = None
        raw_size = None
        num_patches = None
        dark_frames = []
        alpha_offset = None
        beta_offset = None
        global_alignments = []
        local_alignments = []
        section = None

        for _i, line in enumerate(lines):
            if line.startswith("# AreTomo Alignment"):
                header = line
                continue
            elif line.startswith("# RawSize"):
                raw_size = tuple(map(int, line.split("=")[1].split()))
                continue
            elif line.startswith("# NumPatches"):
                num_patches = int(line.split("=")[1])
                continue
            elif line.startswith("# DarkFrame"):
                dark_frames.append(DarkFrameInfo.from_string(line))
                continue
            elif line.startswith("# AlphaOffset"):
                alpha_offset = float(line.split("=")[1])
                continue
            elif line.startswith("# BetaOffset"):
                beta_offset = float(line.split("=")[1])
                continue
            elif line.startswith("# SEC"):
                section = "GlobalAlignment"
                continue
            elif line.startswith("# Local Alignment"):
                section = "LocalAlignment"
                continue

            if section == "GlobalAlignment":
                global_alignments.append(GlobalAlignmentInfo.from_string(line))
            elif section == "LocalAlignment":
                local_alignments.append(LocalAlignmentInfo.from_string(line))

        return cls(
            header=header,
            RawSize=raw_size,
            NumPatches=num_patches,
            DarkFrames=dark_frames,
            AlphaOffset=alpha_offset,
            BetaOffset=beta_offset,
            GlobalAlignments=global_alignments,
            LocalAlignments=local_alignments,
        )

    def __str__(self) -> str:
        dark_frames = "\n".join(map(str, self.DarkFrames))
        global_alignments = "\n".join(map(str, self.GlobalAlignments))
        local_alignments = "" if self.LocalAlignments is None else "\n".join(map(str, self.LocalAlignments))
        return (
            f"{self.header}\n"
            f"# RawSize = {self.RawSize[0]} {self.RawSize[1]} {self.RawSize[2]}\n"
            f"# NumPatches = {self.NumPatches}\n"
            f"{dark_frames}\n"
            f"# AlphaOffset ={self.AlphaOffset:>9.2f}\n"
            f"# BetaOffset ={self.BetaOffset:>9.2f}\n"
            "# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n"
            f"{global_alignments}\n"
            "# Local Alignment\n"
            f"{local_alignments}\n"
        )

    def get_global_alignments(
        self,
        kind: str = "numpy",
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Get the global alignments as a numpy array or pandas DataFrame.

        Args:
            kind (str): Type of the output. Must be "numpy" or "pandas".

        Returns:
            Union[np.ndarray, pd.DataFrame]: Global alignments as a numpy array or pandas DataFrame. In case of numpy,
                the shape of the array will be (len(self.GlobalAlignments), 10). In case of pandas, the DataFrame will
                have 10 columns with the names of the fields in GlobalAlignmentInfo.
        """
        if kind == "numpy":
            return np.array([list(ga) for ga in self.GlobalAlignments])
        elif kind == "pandas":
            return pd.DataFrame(
                [list(ga) for ga in self.GlobalAlignments],
                columns=list(GlobalAlignmentInfo.model_fields.keys()),
            )

    def set_global_alignments(self, value: Union[np.ndarray, pd.DataFrame]):
        """
        Set the global alignments from a numpy array or pandas DataFrame.

        Args:
            value (Union[np.ndarray, pd.DataFrame]): Global alignments as a numpy array or pandas DataFrame.
        """
        global_alignments = []
        if isinstance(value, np.ndarray):
            assert value.shape[1] == 10, "Global alignment must have 10 columns."

            for row in value:
                global_alignments.append(
                    GlobalAlignmentInfo(**dict(zip(GlobalAlignmentInfo.model_fields, row))),
                )

        elif isinstance(value, pd.DataFrame):
            global_alignments = []
            for _, row in value.iterrows():
                global_alignments.append(
                    GlobalAlignmentInfo(**{v: row[v] for v in GlobalAlignmentInfo.model_fields}),
                )
        else:
            raise ValueError("Invalid value type. Must be numpy.ndarray or pandas.DataFrame")

        self.GlobalAlignments = global_alignments

    def get_local_alignments(
        self,
        kind: str = "numpy",
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Get the local alignments as a numpy array or pandas DataFrame.

        Args:
            kind (str): Type of the output. Must be "numpy" or "pandas".

        Returns:
            Union[np.ndarray, pd.DataFrame]: Local alignments as a numpy array or pandas DataFrame. In case of numpy,
                the shape of the array will be (len(self.LocalAlignments), 7). In case of pandas, the DataFrame will
                have 7 columns with the names of the fields in LocalAlignmentInfo.
        """
        if kind == "numpy":
            return np.array([list(la) for la in self.LocalAlignments])
        elif kind == "pandas":
            return pd.DataFrame(
                [list(la) for la in self.LocalAlignments],
                columns=list(LocalAlignmentInfo.model_fields.keys()),
            )

    def set_local_alignments(self, values: Union[np.ndarray, pd.DataFrame]):
        """
        Set the local alignments from a numpy array or pandas DataFrame.

        Args:
            values (Union[np.ndarray, pd.DataFrame]): Local alignments as a numpy array or pandas DataFrame.
        """
        local_alignments = []
        if isinstance(values, np.ndarray):
            assert values.shape[1] == 7, "Local alignment must have 7 columns."

            for row in values:
                local_alignments.append(
                    LocalAlignmentInfo(**dict(zip(LocalAlignmentInfo.model_fields, row))),
                )

        elif isinstance(values, pd.DataFrame):
            local_alignments = []
            for _, row in values.iterrows():
                local_alignments.append(LocalAlignmentInfo(**{v: row[v] for v in LocalAlignmentInfo.model_fields}))
        else:
            raise ValueError("Invalid value type. Must be numpy.ndarray or pandas.DataFrame")

        self.LocalAlignments = local_alignments

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the global and local alignments as numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Global and local alignments as numpy arrays.
        """
        return self.get_global_alignments("numpy"), self.get_local_alignments("numpy")

    def pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the global and local alignments as pandas DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Global and local alignments as pandas DataFrames
        """
        return self.get_global_alignments("pandas"), self.get_local_alignments("pandas")
