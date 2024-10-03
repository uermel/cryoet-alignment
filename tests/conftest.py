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
    res = AreTomo3ALN(
        header="# AreTomo Alignment / Priims bprmMn",
        RawSize=(2032, 2032, 90),
        NumPatches=16,
        DarkFrames=[
            DarkFrameInfo(section_idx=0, val2=0, angle=-66.00),
            DarkFrameInfo(section_idx=1, val2=0, angle=-64.50),
            DarkFrameInfo(section_idx=88, val2=0, angle=66.00),
            DarkFrameInfo(section_idx=89, val2=0, angle=67.50),
        ],
        AlphaOffset=0.00,
        BetaOffset=0.00,
        GlobalAlignments=[
            GlobalAlignmentInfo(
                sec=0,
                rot=-12.6611,
                gmag=1.00000,
                tx=24.786,
                ty=-2.677,
                smean=1.00,
                sfit=1.00,
                scale=1.00,
                base=0.00,
                tilt=-61.50,
            ),
            GlobalAlignmentInfo(
                sec=1,
                rot=-12.6611,
                gmag=1.00000,
                tx=34.451,
                ty=-8.599,
                smean=1.00,
                sfit=1.00,
                scale=1.00,
                base=0.00,
                tilt=-60.00,
            ),
            GlobalAlignmentInfo(
                sec=2,
                rot=-12.6611,
                gmag=1.00000,
                tx=9.951,
                ty=-7.690,
                smean=1.00,
                sfit=1.00,
                scale=1.00,
                base=0.00,
                tilt=-58.50,
            ),
            GlobalAlignmentInfo(
                sec=3,
                rot=-12.6611,
                gmag=1.00000,
                tx=5.538,
                ty=-2.504,
                smean=1.00,
                sfit=1.00,
                scale=1.00,
                base=0.00,
                tilt=-57.00,
            ),
        ],
        LocalAlignments=[
            LocalAlignmentInfo(
                sec_idx=0,
                patch_idx=0,
                center_x=-558.42,
                center_y=-802.00,
                shift_x=-100.07,
                shift_y=24.43,
                is_reliable=1.0,
            ),
            LocalAlignmentInfo(
                sec_idx=0,
                patch_idx=1,
                center_x=-274.86,
                center_y=-757.91,
                shift_x=-36.95,
                shift_y=11.98,
                is_reliable=1.0,
            ),
            LocalAlignmentInfo(
                sec_idx=0,
                patch_idx=2,
                center_x=19.88,
                center_y=-701.35,
                shift_x=29.18,
                shift_y=-5.75,
                is_reliable=1.0,
            ),
            LocalAlignmentInfo(
                sec_idx=0,
                patch_idx=3,
                center_x=233.10,
                center_y=-680.25,
                shift_x=55.29,
                shift_y=-14.05,
                is_reliable=1.0,
            ),
            LocalAlignmentInfo(
                sec_idx=0,
                patch_idx=4,
                center_x=-485.74,
                center_y=-308.27,
                shift_x=-33.72,
                shift_y=10.06,
                is_reliable=1.0,
            ),
            LocalAlignmentInfo(
                sec_idx=0,
                patch_idx=5,
                center_x=-201.52,
                center_y=-264.31,
                shift_x=-10.65,
                shift_y=2.03,
                is_reliable=1.0,
            ),
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
