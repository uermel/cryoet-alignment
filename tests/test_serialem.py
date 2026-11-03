"""Tests for the SerialEM tilt-series .mdoc reader and writer."""

from pathlib import Path

import pytest
from cryoet_alignment import read, write
from cryoet_alignment.io.serialem import MdocSection, SerialEMMdoc

DATA = Path(__file__).parent / "data" / "serialem" / "TS_01.mrc.mdoc"
ANGLES = [0.0, 3.0, -3.0, -6.0, 6.0, 9.0, -9.0]


def test_header_titles_and_sections():
    mdoc = read(DATA)
    assert isinstance(mdoc, SerialEMMdoc)
    assert mdoc.pixel_spacing == 1.54
    assert mdoc.voltage == 300.0
    assert mdoc.image_file == "TS_01.mrc"
    assert mdoc.image_size == (4096, 4096)
    assert mdoc.data_mode == 1
    assert mdoc.tilt_axis_angle == 85.3
    assert mdoc.n_sections == 7
    assert mdoc.tilt_angles == pytest.approx([a + 0.00146 for a in ANGLES])

    first = mdoc.sections[0]
    assert first.z_value == 0
    assert first.exposure_dose == 3.10
    assert first.prior_record_dose == 0.0
    assert first.defocus == -2.8
    assert first.target_defocus == -3.5
    assert first.num_sub_frames == 8
    assert first.magnification == 81000
    assert first.sub_frame_path == "X:\\DoseFractions\\TS_01\\TS_01_000_0.0.eer"
    assert first.sub_frame_name == "TS_01_000_0.0.eer"
    assert first.extra["StageZ"] == "-88.7476"
    assert first.extra["ImageShift"] == "0.0244376 0.0315099"


def test_acquisition_order_follows_date_time_then_z_value():
    mdoc = read(DATA)
    assert mdoc.acquisition_order() == list(range(7))

    # A stack sorted by angle: DateTime still gives the acquisition order.
    by_angle = sorted(mdoc.sections, key=lambda s: s.tilt_angle)
    resorted = SerialEMMdoc(
        sections=[s.model_copy(update={"z_value": z}) for z, s in enumerate(by_angle)],
    )
    expected = [ANGLES.index(round(s.tilt_angle - 0.00146, 3)) for s in by_angle]
    assert resorted.acquisition_order() == expected

    # Without DateTime the ZValue order is the acquisition order.
    undated = SerialEMMdoc(
        sections=[s.model_copy(update={"date_time": None}) for s in resorted.sections],
    )
    assert undated.acquisition_order() == list(range(7))


def test_pre_exposure_prefers_prior_record_dose():
    mdoc = read(DATA)
    pre = mdoc.pre_exposure()
    assert pre == [s.prior_record_dose for s in mdoc.sections]
    assert pre[0] == 0.0 and pre[1] == pytest.approx(3.10)


def test_pre_exposure_sums_exposure_dose_in_acquisition_order():
    mdoc = read(DATA)
    by_angle = sorted(mdoc.sections, key=lambda s: s.tilt_angle)
    stripped = SerialEMMdoc(
        sections=[s.model_copy(update={"z_value": z, "prior_record_dose": None}) for z, s in enumerate(by_angle)],
    )
    pre = stripped.pre_exposure()
    # Each image's pre-exposure is the dose of the images acquired before it, not including itself.
    for section, value in zip(by_angle, pre):
        assert value == pytest.approx(section.prior_record_dose)


def test_pre_exposure_uniform_override_and_uncalibrated_dose():
    mdoc = read(DATA)
    assert mdoc.pre_exposure(dose_per_tilt=2.0) == [2.0 * i for i in range(7)]
    uncalibrated = SerialEMMdoc(
        sections=[s.model_copy(update={"prior_record_dose": 0.0, "exposure_dose": 0.0}) for s in mdoc.sections],
    )
    assert uncalibrated.pre_exposure() is None


def test_round_trip(tmp_path):
    mdoc = read(DATA)
    out = tmp_path / "out.mdoc"
    write(mdoc, out)
    again = read(out)
    assert again == mdoc


def test_refuses_montages_unknown_sections_and_missing_angles():
    with pytest.raises(ValueError, match="montage"):
        SerialEMMdoc.from_string("PixelSpacing = 1\n[MontSection = 0]\nTiltAngle = 0\n")
    with pytest.raises(ValueError, match="unsupported mdoc section"):
        SerialEMMdoc.from_string("[FrameSet = 0]\nTiltAngle = 0\n")
    with pytest.raises(ValueError, match="TiltAngle"):
        SerialEMMdoc.from_string("[ZValue = 0]\nExposureDose = 1\n")
    with pytest.raises(ValueError, match="more than once"):
        SerialEMMdoc(sections=[MdocSection(z_value=0, tilt_angle=0.0), MdocSection(z_value=0, tilt_angle=3.0)])
    with pytest.raises(ValueError, match="no \\[ZValue"):
        SerialEMMdoc.from_string("PixelSpacing = 1\n")


def test_unparseable_date_time_falls_back_to_z_value_order():
    mdoc = SerialEMMdoc(
        sections=[
            MdocSection(z_value=0, tilt_angle=0.0, date_time="02-Jul-21  10:09:00"),
            MdocSection(z_value=1, tilt_angle=3.0, date_time="not a date"),
        ],
    )
    assert mdoc.sections[1].acquired_at is None
    assert mdoc.acquisition_order() == [0, 1]
