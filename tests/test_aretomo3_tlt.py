"""Tests for the AreTomo3 _TLT.txt model."""

from pathlib import Path

import pytest
from cryoet_alignment import read, write
from cryoet_alignment.io.aretomo3 import AreTomo3TLT

DATA_DIR = Path(__file__).parent / "data"


def test_parse_real_fixture():
    """Genuine AreTomo3 2.3.1 output (24jul16a run, -TotalDose 0): 31 raw tilts, the
    acquisition column is a permutation of 1..31 and the dose column is the
    tilt-attenuated mdoc ExposureDose."""
    tlt = AreTomo3TLT.from_file(DATA_DIR / "test_TLT.txt")
    assert tlt.n_rows == 31
    assert tlt.has_acq_index and tlt.has_dose
    assert tlt.rows[0].tilt == -45.01
    assert tlt.rows[0].acq_index == 31
    assert tlt.rows[0].dose == 1.49
    assert sorted(tlt.acq_indices) == list(range(1, 32))
    assert tlt.tilts == sorted(tlt.tilts)


def test_string_roundtrip_is_byte_identical_to_aretomo3():
    """Our serializer reproduces AreTomo3's "%8.2f  %4d  %8.2f" lines byte for byte."""
    path = DATA_DIR / "test_TLT.txt"
    tlt = AreTomo3TLT.from_file(path)
    assert str(tlt) == path.read_text()
    assert str(AreTomo3TLT.from_string(str(tlt))) == str(tlt)


def test_one_and_two_column_files():
    """AreTomo3's loader parses ``%f %d %f`` and accepts 1-3 items per line."""
    one = AreTomo3TLT.from_string("-60.0\n-57.0\n\n-54.0\n")
    assert one.n_rows == 3
    assert not one.has_acq_index and not one.has_dose
    assert one.acq_indices is None and one.doses is None
    assert str(one) == "  -60.00\n  -57.00\n  -54.00\n"

    two = AreTomo3TLT.from_string("-60.0 3\n-57.0 1\n-54.0 2\n")
    assert two.acq_indices == [3, 1, 2]
    assert not two.has_dose
    assert str(two) == "  -60.00     3\n  -57.00     1\n  -54.00     2\n"


def test_zero_based_indices_are_normalized():
    """AreTomo3's writer converts 0-based acquisition indices to 1-based
    (CTsPackage.cpp mSaveTiltFile); the model does the same on construction."""
    tlt = AreTomo3TLT.from_string("-60.0 2 3.0\n-57.0 0 3.0\n-54.0 1 3.0\n")
    assert tlt.acq_indices == [3, 1, 2]


@pytest.mark.parametrize(
    "text",
    [
        "",  # no rows
        "-60.0 1 3.0\n-57.0 2\n",  # mixed column counts
        "-60.0 1 3.0\n-57.0 1 3.0\n",  # duplicate acquisition index
        "-60.0 2 3.0\n-57.0 3 3.0\n",  # indices not starting at 0/1
        "-60.0 1 -3.0\n-57.0 2 3.0\n",  # negative dose
        "nan 1 3.0\n-57.0 2 3.0\n",  # non-finite tilt
    ],
)
def test_malformed_files_are_rejected(text):
    with pytest.raises(ValueError):
        AreTomo3TLT.from_string(text)


def test_read_write_api(tmp_path):
    tlt = read(DATA_DIR / "test_TLT.txt", reader="aretomo3_tlt")
    assert isinstance(tlt, AreTomo3TLT)
    out = tmp_path / "out_TLT.txt"
    write(tlt, out)
    assert out.read_text() == (DATA_DIR / "test_TLT.txt").read_text()
