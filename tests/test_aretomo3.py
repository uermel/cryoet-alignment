from pathlib import Path
from typing import Tuple

from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN


def test_aln(aln_file: Tuple[Path, AreTomo3ALN]):
    path, exp = aln_file

    aln = AreTomo3ALN.from_file(path)

    for key in exp.model_fields:
        assert getattr(aln, key) == getattr(exp, key), f"Field {key} does not match."

    with open(path, "r") as f:
        assert str(aln) == f.read(), "Serialization does not match."
