"""Tests for the AreTomo3 _CTF.txt model."""

from pathlib import Path

import pytest
from cryoet_alignment import read, write
from cryoet_alignment.io.aretomo3 import AreTomo3CTF

DATA_DIR = Path(__file__).parent / "data"


def test_parse_7col_fixture():
    """CTFFIND4-style 7-column file (no dfHand)."""
    ctf = AreTomo3CTF.from_file(DATA_DIR / "test_CTF.txt")
    assert ctf.n_rows == 3
    assert ctf.rows[0].df_max_a == 25951.41
    assert ctf.rows[0].df_min_a == 24469.37
    assert ctf.rows[2].azimuth_deg == 77.32
    assert all(r.df_hand is None for r in ctf.rows)


def test_parse_8col_real_fixture():
    """Genuine AreTomo3 (-kV 300) output from a 24jul16a run: 31 raw tilts,
    dfHand column present (and -1 — the +1 normalization is conditional)."""
    ctf = AreTomo3CTF.from_file(DATA_DIR / "test_8col_CTF.txt")
    assert ctf.n_rows == 31
    assert ctf.rows[0].df_max_a == 21717.31
    assert ctf.rows[0].df_min_a == 20740.78
    assert all(r.df_hand == -1 for r in ctf.rows)
    assert all(r.df_max_a >= r.df_min_a for r in ctf.rows)

    # data round trip through the serializer (headers/spacing are normalized,
    # so compare values, not bytes)
    ctf_rt = AreTomo3CTF.from_string(str(ctf))
    assert ctf_rt.n_rows == ctf.n_rows
    for a, b in zip(ctf.rows, ctf_rt.rows):
        assert a == b


def test_string_roundtrip_exact():
    """Our own serialization is byte-stable through a parse cycle."""
    ctf = AreTomo3CTF.from_file(DATA_DIR / "test_8col_CTF.txt")
    once = str(ctf)
    assert str(AreTomo3CTF.from_string(once)) == once


def test_zero_and_one_based_numbering():
    """AreTomo3's loader min-normalizes the micrograph column, so 0-based and
    1-based numbering are both accepted (CLoadCtfResults.cpp:84-94)."""
    one_based = "1 21000.5 20000.2 12.0 0.0 0.15 4.5 1\n2 22000.0 21500.0 170.0 0.0 0.2 5.0 1\n"
    zero_based = "0 21000.5 20000.2 12.0 0.0 0.15 4.5 1\n1 22000.0 21500.0 170.0 0.0 0.2 5.0 1\n"
    f1 = AreTomo3CTF.from_string(one_based)
    f0 = AreTomo3CTF.from_string(zero_based)
    assert f1.rows[0].df_max_a == f0.rows[0].df_max_a == 21000.5
    assert str(f1) == str(f0)  # serialization renumbers 1..N either way


def test_out_of_order_rows_are_resorted():
    text = "2 22000.0 21500.0 170.0 0.0 0.2 5.0 1\n1 21000.5 20000.2 12.0 0.0 0.15 4.5 1\n"
    ctf = AreTomo3CTF.from_string(text)
    assert ctf.rows[0].df_max_a == 21000.5  # rows[i] == raw-stack ordinal i
    assert ctf.rows[1].df_max_a == 22000.0


def test_rejects_duplicates_gaps_and_bad_lines():
    with pytest.raises(ValueError, match="contiguous unique"):
        AreTomo3CTF.from_string("1 1 1 1 0 0 1 1\n1 2 2 2 0 0 1 1\n")  # duplicate
    with pytest.raises(ValueError, match="contiguous unique"):
        AreTomo3CTF.from_string("1 1 1 1 0 0 1 1\n3 2 2 2 0 0 1 1\n")  # gap
    with pytest.raises(ValueError, match="7 or 8 columns"):
        AreTomo3CTF.from_string("1 2 3\n")
    with pytest.raises(ValueError, match="no data rows"):
        AreTomo3CTF.from_string("# only a header\n")
    with pytest.raises(ValueError, match="non-finite"):
        AreTomo3CTF.from_string("1 nan 1 1 0 0 1 1\n")


def test_api_read_write(tmp_path):
    """read(reader='aretomo3_ctf') and write() with type inference."""
    ctf = read(DATA_DIR / "test_8col_CTF.txt", reader="aretomo3_ctf")
    assert isinstance(ctf, AreTomo3CTF)

    out = tmp_path / "out_CTF.txt"
    write(ctf, out)
    ctf_rt = read(out, reader="aretomo3_ctf")
    assert ctf_rt.n_rows == ctf.n_rows
    for a, b in zip(ctf.rows, ctf_rt.rows):
        assert a == b
