from pathlib import Path
from typing import Tuple

from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN


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
