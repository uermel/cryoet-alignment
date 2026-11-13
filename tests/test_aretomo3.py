from pathlib import Path
from typing import Tuple

import pytest
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN, DarkFrameInfo, GlobalAlignmentInfo


def test_aln(aln_file: Tuple[Path, AreTomo3ALN]):
    path, exp = aln_file

    aln = AreTomo3ALN.from_file(path)

    for key in exp.model_fields:
        assert getattr(aln, key) == getattr(exp, key), f"Field {key} does not match."

    assert aln.Thickness is None

    with open(path, "r") as f:
        assert str(aln) == f.read(), "Serialization does not match."


def test_aln_thickness(aln_file: Tuple[Path, AreTomo3ALN]):
    """AreTomo3 emits '# Thickness = <px>' before the SEC table; files with and
    without the line must both round-trip byte-identically."""
    plain_path, exp = aln_file
    path = plain_path.parent / "test_thickness.aln"

    aln = AreTomo3ALN.from_file(path)

    assert aln.Thickness == 1240
    for key in exp.model_fields:
        if key == "Thickness":
            continue
        assert getattr(aln, key) == getattr(exp, key), f"Field {key} does not match."

    with open(path, "r") as f:
        assert str(aln) == f.read(), "Serialization does not match."


LEGACY_PERMUTED = Path(__file__).parent / "data" / "test_legacy_permuted.aln"


def test_aln_legacy_permuted_sec():
    """Legacy AreTomo files list rows by tilt angle with SEC = 0-based index into the input stack. For a stack in
    acquisition order (here dose-symmetric 0, +3, -3, +6, -6, +9, -9) SEC is a non-ascending permutation."""
    aln = AreTomo3ALN.from_file(LEGACY_PERMUTED)

    assert aln.sec_base == 0
    assert aln.z_indices() == [6, 4, 2, 0, 1, 3, 5]
    assert aln.raw_tilts() == [0.0, 3.0, -3.0, 6.0, -6.0, 9.0, -9.0]
    assert aln.row_slots() == [3, 4, 2, 5, 1, 6, 0]

    with open(LEGACY_PERMUTED, "r") as f:
        assert str(aln) == f.read(), "Serialization does not match."


def test_aln_legacy_permuted_sec_with_dark_frame():
    """A permuted 0-based SEC still resolves alongside dark frames (legacy files carry 0 in val2)."""
    rows = [(4, -6.0), (2, -3.0), (0, 0.0), (1, 3.0), (3, 6.0)]
    aln = AreTomo3ALN(
        RawSize=(1024, 1024, 6),
        NumPatches=0,
        DarkFrames=[DarkFrameInfo(section_idx=5, val2=0, angle=9.0)],
        GlobalAlignments=[GlobalAlignmentInfo(sec=s, rot=0.0, tx=0.0, ty=0.0, tilt=t) for s, t in rows],
    )

    assert aln.sec_base == 0
    assert aln.z_indices() == [4, 2, 0, 1, 3]
    assert aln.raw_tilts() == [0.0, 3.0, -3.0, 6.0, -6.0, 9.0]


def test_aln_repeated_sec_raises():
    rows = [(1, -3.0), (2, 0.0), (2, 3.0)]
    with pytest.raises(ValueError, match="repeated sections"):
        AreTomo3ALN(
            RawSize=(1024, 1024, 3),
            NumPatches=0,
            DarkFrames=[],
            GlobalAlignments=[GlobalAlignmentInfo(sec=s, rot=0.0, tx=0.0, ty=0.0, tilt=t) for s, t in rows],
        )


def test_aln_legacy_permuted_sec_dark_frame_input_index():
    """Legacy AreTomo 1.x files write "DarkFrame = <tilt-sorted index> <input-stack index> <angle>" with SEC in
    input-stack order; the dark frame then sits at its second value (here the +6 image of a bi-directional stack)."""
    path = Path(__file__).parent / "data" / "test_legacy_permuted_dark.aln"
    aln = AreTomo3ALN.from_file(path)

    assert aln.sec_base == 0
    assert aln.z_indices() == [6, 5, 4, 0, 1, 3]
    assert aln.dark_indices() == [2]
    assert aln.raw_tilts() == [0.0, 3.0, 6.0, 9.0, -3.0, -6.0, -9.0]
    assert aln.row_slots() == [3, 4, -1, 5, 2, 1, 0]

    with open(path, "r") as f:
        assert str(aln) == f.read(), "Serialization does not match."
