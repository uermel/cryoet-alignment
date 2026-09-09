from pathlib import Path
from typing import Tuple

import pytest
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN, DarkFrameInfo, GlobalAlignmentInfo, LocalAlignmentInfo
from cryoet_alignment.io.imod.newst import ImodNEWSTCOM
from cryoet_alignment.io.imod.rawtlt import ImodRAWTLT
from cryoet_alignment.io.imod.tilt import ImodTILTCOM
from cryoet_alignment.io.imod.xf import ImodXF, ImodXFInfo


# Read AreTomo3
@pytest.fixture
def aln_file() -> Tuple[Path, AreTomo3ALN]:
    """A small but self-consistent AreTomo3 .aln: RawSize z = 8, dark frames at raw
    sections 1 (SEC 2) and 6 (SEC 7), six global rows with SEC 1,3,4,5,6,8 and
    NumPatches = 2 (12 local rows over the dark-removed list)."""
    globals_ = [
        (1, 24.786, -2.677, -67.50),
        (3, 34.451, -8.599, -64.50),
        (4, 9.951, -7.690, -63.00),
        (5, 5.538, -2.504, -61.50),
        (6, -3.120, 1.877, -60.00),
        (8, 7.004, -0.331, 67.50),
    ]
    vals = [(-558.42, -802.00, -100.07, 24.43), (-274.86, -757.91, -36.95, 11.98)]
    res = AreTomo3ALN(
        header="# AreTomo Alignment / Priims bprmMn",
        RawSize=(2032, 2032, 8),
        NumPatches=2,
        DarkFrames=[
            DarkFrameInfo(section_idx=1, val2=2, angle=-66.00),
            DarkFrameInfo(section_idx=6, val2=7, angle=66.00),
        ],
        AlphaOffset=0.00,
        BetaOffset=0.00,
        GlobalAlignments=[
            GlobalAlignmentInfo(
                sec=sec,
                rot=-12.6611,
                gmag=1.0,
                tx=tx,
                ty=ty,
                smean=1.0,
                sfit=1.0,
                scale=1.0,
                base=0.0,
                tilt=tilt,
            )
            for sec, tx, ty, tilt in globals_
        ],
        LocalAlignments=[
            LocalAlignmentInfo(
                sec_idx=s,
                patch_idx=p,
                center_x=cx + s,
                center_y=cy,
                shift_x=sx,
                shift_y=sy,
                is_reliable=1.0,
            )
            for s in range(6)
            for p, (cx, cy, sx, sy) in enumerate(vals)
        ],
    )
    return Path(__file__).parent / "data" / "test.aln", res


# Read IMOD
@pytest.fixture
def newstcom_file() -> Tuple[Path, ImodNEWSTCOM]:
    res = ImodNEWSTCOM(
        AntialiasFilter=-1,
        InputFile="mba2012-02-01-1.mrc",
        OutputFile="mba2012-02-01-1_ali.mrc",
        TransformFile="mba2012-02-01-1.xf",
        TaperAtFill=(1, 1),
        AdjustOrigin=True,
        OffsetsInXandY=(0.0, 0.0),
        ImagesAreBinned=1.0,
        BinByFactor=2,
    )
    return Path(__file__).parent / "data" / "newst.com", res


@pytest.fixture
def tiltcom_file() -> Tuple[Path, ImodTILTCOM]:
    res = ImodTILTCOM(
        InputProjections="mba2012-02-01-1_ali.mrc",
        OutputFile="mba2012-02-01-1_full_rec.mrc",
        IMAGEBINNED=2,
        TILTFILE="mba2012-02-01-1.tlt",
        THICKNESS=900,
        RADIAL=(0.35, 0.035),
        FalloffIsTrueSigma=1,
        XAXISTILT=0.0,
        SCALE=(0.0, 0.1),
        PERPENDICULAR=True,
        MODE=2,
        FULLIMAGE=(2032, 2032),
        SUBSETSTART=(0, 0),
        AdjustOrigin=True,
        ActionIfGPUFails=(1, 2),
        XTILTFILE="mba2012-02-01-1.xtilt",
        OFFSET=0.0,
        SHIFT=(0.0, 0.0),
    )
    return Path(__file__).parent / "data" / "tilt.com", res


@pytest.fixture
def rawtlt_file() -> Tuple[Path, ImodRAWTLT]:
    res = ImodRAWTLT(
        angles=[-66.0, -64.5, -63.0],
    )
    return Path(__file__).parent / "data" / "test.rawtlt", res


@pytest.fixture
def xf_file() -> Tuple[Path, ImodXF]:
    res = ImodXF(
        alignments=[
            ImodXFInfo(mxx=0.9803519, mxy=-0.1972494, myx=0.1972494, myy=0.9803519, sx=22.751, sy=-0.799),
            ImodXFInfo(mxx=0.9803793, mxy=-0.1979111, myx=0.1979111, myy=0.9803793, sx=6.676, sy=1.969),
            ImodXFInfo(mxx=0.9796340, mxy=-0.1974376, myx=0.1974376, myy=0.9796341, sx=13.633, sy=-3.543),
        ],
    )
    return Path(__file__).parent / "data" / "test.xf", res


# Convert IMOD -> CDP
@pytest.fixture
def imod_base() -> str:
    return str(Path(__file__).parent / "data" / "convert" / "imod_1" / "tilt_1")
