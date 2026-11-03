"""Tests for the Warp .tomostar and .settings models."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cryoet_alignment import read, write
from cryoet_alignment.io.warp import TomostarRow, WarpSettings, WarpTomostar

DATA = Path(__file__).parent / "data" / "warp"


def test_tomostar_real_fixture_roundtrip():
    t = WarpTomostar.from_file(DATA / "test.tomostar")
    assert t.n_rows == 31
    r = t.rows[0]
    assert r.movie_name.endswith("24jul16a_Position_16_3_031_-45.00_20240716_210415_EER.eer")
    assert (r.angle_tilt, r.axis_angle, r.dose) == (45.01, 83.94, 116.1)
    assert r.average_intensity == pytest.approx(14.586) and r.masked_fraction == 0.0
    again = WarpTomostar.from_string(str(t))
    assert again.movie_names == t.movie_names
    assert [x.angle_tilt for x in again.rows] == [x.angle_tilt for x in t.rows]
    assert [x.dose for x in again.rows] == pytest.approx([x.dose for x in t.rows])
    assert str(WarpTomostar.from_string(str(t))) == str(t)  # our serialization is stable


def test_tomostar_minimal_and_extra_columns():
    t = WarpTomostar(
        rows=[
            TomostarRow(movie_name="../frames/a.mrc", angle_tilt=-30.0, axis_angle=85.0, dose=0.0),
            TomostarRow(movie_name="../frames/b.mrc", angle_tilt=0.0, axis_angle=85.0, dose=3.0),
        ],
    )
    text = str(t)
    assert "_wrpMovieName #1" in text and "_wrpAverageIntensity" not in text
    assert WarpTomostar.from_string(text).rows[1].dose == 3.0
    extra = "\ndata_\n\nloop_\n_wrpMovieName #1\n_wrpAngleTilt #2\n_wrpDose #3\n_wrpFoo #4\n  x.mrc  10.0  0.0  bar\n"
    t2 = WarpTomostar.from_string(extra)
    assert t2.rows[0].extra == {"_wrpFoo": "bar"} and t2.rows[0].axis_angle == 0.0
    assert "_wrpFoo #5" in str(t2) and str(t2).rstrip().endswith("bar")
    with pytest.raises(ValueError, match="empty wrpMovieName"):
        WarpTomostar(rows=[TomostarRow(movie_name="", angle_tilt=0.0)])


def test_settings_parse_and_properties():
    s = WarpSettings.from_file(DATA / "test.settings")
    assert s.params["ProcessCTF"] == "True"
    assert s.pixel_size_a == 1.54 and s.data_folder == "tomostar" and s.processing_folder == "warp_tiltseries"
    assert s.extension == "*.tomostar" and s.exposure_per_tilt == 3.87
    assert s.tomo_dims_px == [4096, 4096, 2000]
    assert (s.voltage_kv, s.cs_mm, s.amplitude_contrast) == (300.0, 2.7, 0.07)
    again = WarpSettings.from_string(str(s))
    assert again.params == s.params and again.sections == s.sections
    assert str(s).startswith('<?xml version="1.0" encoding="utf-8"?>\n<Settings>\n\t<Param Name="ProcessCTF"')


def test_settings_create_matches_real_file_except_the_create_settings_slots():
    real = WarpSettings.from_file(DATA / "test.settings")
    made = WarpSettings.create(
        pixel_size_a=1.54,
        exposure_per_tilt=3.87,
        tomo_dims_px=(4096, 4096, 2000),
        gain_path="24jul16a_gain.gain",
        eer_group_frames=40,
    )
    assert made.params == real.params
    for section in real.sections:
        assert made.sections[section] == real.sections[section], section
    # differing inputs land in exactly the create_settings slots
    other = WarpSettings.create(
        pixel_size_a=2.0,
        exposure_per_tilt=3.0,
        tomo_dims_px=(100, 200, 50),
        data_folder="d",
        processing_folder="p",
        extension="*.x",
    )
    diff = {(sec, k) for sec in real.sections for k, v in other.sections[sec].items() if real.sections[sec][k] != v}
    assert diff == {
        ("Import", "PixelSize"),
        ("Import", "DosePerAngstromFrame"),
        ("Tomo", "DimensionsX"),
        ("Tomo", "DimensionsY"),
        ("Tomo", "DimensionsZ"),
        ("Import", "DataFolder"),
        ("Import", "ProcessingFolder"),
        ("Import", "Extension"),
        ("Import", "GainPath"),
        ("Import", "CorrectGain"),
    }
    assert (
        other.sections["Import"]["DosePerAngstromFrame"] == "-3" and other.sections["Import"]["CorrectGain"] == "False"
    )


def test_read_write_api(tmp_path):
    t = read(DATA / "test.tomostar")
    assert isinstance(t, WarpTomostar)
    write(t, tmp_path / "x.tomostar")
    assert isinstance(read(tmp_path / "x.tomostar"), WarpTomostar)
    s = read(DATA / "test.settings")
    assert isinstance(s, WarpSettings)
    write(s, tmp_path / "x.settings")
    assert WarpSettings.from_file(tmp_path / "x.settings").sections == s.sections


_WARPTOOLS = os.environ.get("CRYOET_ALIGNMENT_WARPTOOLS_BIN") or shutil.which("WarpTools")


@pytest.mark.skipif(_WARPTOOLS is None, reason="WarpTools not available")
def test_settings_create_parity_with_warptools(tmp_path):
    subprocess.run(
        [
            _WARPTOOLS,
            "create_settings",
            "-o",
            str(tmp_path / "w.settings"),
            "--folder_data",
            "tomostar",
            "--folder_processing",
            "warp_tiltseries",
            "--extension",
            "*.tomostar",
            "--angpix",
            "1.54",
            "--exposure",
            "3.87",
            "--tomo_dimensions",
            "4096x4096x2000",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "DOTNET_USE_POLLING_FILE_WATCHER": "1"},
    )
    theirs = WarpSettings.from_file(tmp_path / "w.settings")
    ours = WarpSettings.create(pixel_size_a=1.54, exposure_per_tilt=3.87, tomo_dims_px=(4096, 4096, 2000))
    assert theirs.params == ours.params
    for section in theirs.sections:
        assert theirs.sections[section] == ours.sections[section], section
