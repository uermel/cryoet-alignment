"""Tests for the Warp alignment format.

Covers the ``WarpAlignment`` Pydantic schema, the conversion math in
``Alignment.from_warp`` / ``Alignment.to_warp``, and the native XML I/O
(``from_string``/``from_file``/``__str__``/``to_file``) — all dependency-free.
A final cross-validation test compares the native I/O against ``warpylib``
field-by-field and only runs where warpylib happens to be installed.
"""

from pathlib import Path

import pytest
from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.aretomo3.aln import GlobalAlignmentInfo
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.warp import WarpAlignment, WarpAlignmentEntry


def _make_simple_aln(n_tilts: int = 5, pixel_size_a: float = 2.0) -> AreTomo3ALN:
    """Construct a small AreTomo3ALN with no dark frames for round-trip tests."""
    global_alignments = []
    for i in range(n_tilts):
        global_alignments.append(
            GlobalAlignmentInfo(
                sec=i + 1,
                rot=85.5 + 0.1 * i,  # per-tilt rotation, slightly varying
                gmag=1.0,
                tx=10.0 + 0.5 * i,
                ty=-3.0 + 0.2 * i,
                smean=1.0,
                sfit=1.0,
                scale=1.0,
                base=0.0,
                tilt=-30.0 + 15.0 * i,
            ),
        )

    return AreTomo3ALN(
        RawSize=(512, 512, n_tilts),
        NumPatches=0,
        DarkFrames=[],
        AlphaOffset=0.0,
        BetaOffset=0.0,
        GlobalAlignments=global_alignments,
    )


def test_warp_alignment_construction():
    """WarpAlignment is a plain Pydantic model — construction shouldn't need warpylib."""
    entry = WarpAlignmentEntry(
        z_index=0,
        tilt_angle=-30.0,
        tilt_axis_angle=85.5,
        tilt_axis_offset_x=20.0,
        tilt_axis_offset_y=-6.0,
    )
    warp = WarpAlignment(
        n_tilts=1,
        pixel_size_a=2.0,
        image_dimensions_physical=[1024.0, 1024.0],
        volume_dimensions_physical=[1024.0, 1024.0, 800.0],
        entries=[entry],
    )
    assert warp.n_tilts == 1
    assert warp.entries[0].tilt_angle == -30.0
    assert warp.pixel_size_a == 2.0


def test_aretomo3_to_warp_to_aretomo3_roundtrip():
    """``from_aretomo3 → to_warp → from_warp → to_aretomo`` must preserve the global
    alignment fields (rot, tx, ty, tilt, sec) up to numerical precision."""
    aln = _make_simple_aln(n_tilts=5, pixel_size_a=2.0)
    pixel_size_a = 2.0
    image_size_px = (512, 512)

    # AreTomo → canonical Alignment
    alignment = Alignment.from_aretomo3(aln, vol_size=(512, 512, 400))

    # canonical → Warp
    warp = alignment.to_warp(pixel_size_a=pixel_size_a, image_size_px=image_size_px)

    assert warp.n_tilts == len(aln.GlobalAlignments)
    assert warp.pixel_size_a == pixel_size_a

    # Warp → canonical Alignment
    alignment_rt = Alignment.from_warp(warp, vol_size=(512, 512, 400))

    # canonical → AreTomo
    aln_rt = alignment_rt.to_aretomo(ts_size=(512, 512, 5))

    assert len(aln_rt.GlobalAlignments) == len(aln.GlobalAlignments)
    for orig, rt in zip(aln.GlobalAlignments, aln_rt.GlobalAlignments):
        assert orig.sec == rt.sec
        assert orig.tilt == pytest.approx(rt.tilt, abs=1e-4)
        assert orig.rot == pytest.approx(rt.rot, abs=1e-4)
        assert orig.tx == pytest.approx(rt.tx, abs=1e-4)
        assert orig.ty == pytest.approx(rt.ty, abs=1e-4)


def test_warp_to_alignment_to_warp_roundtrip():
    """``WarpAlignment → from_warp → to_warp`` must preserve all per-tilt fields."""
    entries = [
        WarpAlignmentEntry(
            z_index=i,
            tilt_angle=-30.0 + 15.0 * i,
            tilt_axis_angle=85.5 + 0.1 * i,
            tilt_axis_offset_x=20.0 + 1.0 * i,
            tilt_axis_offset_y=-6.0 + 0.4 * i,
        )
        for i in range(5)
    ]
    pixel_size_a = 2.0
    warp = WarpAlignment(
        n_tilts=len(entries),
        pixel_size_a=pixel_size_a,
        image_dimensions_physical=[1024.0, 1024.0],
        volume_dimensions_physical=[1024.0, 1024.0, 800.0],
        entries=entries,
    )

    alignment = Alignment.from_warp(warp, vol_size=(512, 512, 400))
    warp_rt = alignment.to_warp(pixel_size_a=pixel_size_a, image_size_px=(512, 512))

    assert warp_rt.n_tilts == warp.n_tilts
    for orig, rt in zip(warp.entries, warp_rt.entries):
        assert orig.z_index == rt.z_index
        assert orig.tilt_angle == pytest.approx(rt.tilt_angle, abs=1e-4)
        assert orig.tilt_axis_angle == pytest.approx(rt.tilt_axis_angle, abs=1e-4)
        assert orig.tilt_axis_offset_x == pytest.approx(rt.tilt_axis_offset_x, abs=1e-3)
        assert orig.tilt_axis_offset_y == pytest.approx(rt.tilt_axis_offset_y, abs=1e-3)


def test_to_warp_pixel_unit_conversion():
    """Sanity-check that the px ↔ Å conversion in ``to_warp`` actually scales by
    ``pixel_size_a`` rather than copying the value through."""
    aln = _make_simple_aln(n_tilts=3, pixel_size_a=2.0)
    alignment = Alignment.from_aretomo3(aln, vol_size=(512, 512, 400))

    pixel_size_a = 2.0
    warp = alignment.to_warp(pixel_size_a=pixel_size_a, image_size_px=(512, 512))

    for ali, entry in zip(aln.GlobalAlignments, warp.entries):
        assert entry.tilt_axis_offset_x == pytest.approx(ali.tx * pixel_size_a, abs=1e-6)
        assert entry.tilt_axis_offset_y == pytest.approx(ali.ty * pixel_size_a, abs=1e-6)


def test_from_warp_pixel_unit_conversion():
    """Sanity-check that ``from_warp`` divides Å offsets by ``pixel_size_a``."""
    pixel_size_a = 4.0
    warp = WarpAlignment(
        n_tilts=2,
        pixel_size_a=pixel_size_a,
        image_dimensions_physical=[2048.0, 2048.0],
        volume_dimensions_physical=[2048.0, 2048.0, 1600.0],
        entries=[
            WarpAlignmentEntry(
                z_index=0,
                tilt_angle=0.0,
                tilt_axis_angle=85.5,
                tilt_axis_offset_x=40.0,
                tilt_axis_offset_y=-12.0,
            ),
            WarpAlignmentEntry(
                z_index=1,
                tilt_angle=15.0,
                tilt_axis_angle=85.6,
                tilt_axis_offset_x=44.0,
                tilt_axis_offset_y=-10.4,
            ),
        ],
    )

    alignment = Alignment.from_warp(warp, vol_size=(512, 512, 400))

    assert alignment.per_section_alignment_parameters[0].x_offset == pytest.approx(
        40.0 / pixel_size_a,
        abs=1e-6,
    )
    assert alignment.per_section_alignment_parameters[1].y_offset == pytest.approx(
        -10.4 / pixel_size_a,
        abs=1e-6,
    )


def test_warp_drops_dark_frames_correctly():
    """If the source AreTomo3 .aln has dark frames, they should NOT appear in the
    Warp representation (Warp's per-tilt arrays cover the kept tilts only)."""
    n_kept = 5
    aln = _make_simple_aln(n_tilts=n_kept, pixel_size_a=2.0)
    # Inject one dark frame at z_index=2 — but to keep the AreTomo3ALN consistent we
    # build a fresh one with the dark frame in place.
    from cryoet_alignment.io.aretomo3.aln import DarkFrameInfo

    aln = AreTomo3ALN(
        RawSize=(512, 512, n_kept + 1),
        NumPatches=0,
        DarkFrames=[DarkFrameInfo(section_idx=2, val2=0, angle=0.0)],
        AlphaOffset=0.0,
        BetaOffset=0.0,
        GlobalAlignments=aln.GlobalAlignments,
    )

    alignment = Alignment.from_aretomo3(aln, vol_size=(512, 512, 400))
    warp = alignment.to_warp(pixel_size_a=2.0, image_size_px=(512, 512))
    assert warp.n_tilts == n_kept


# ---------------------------------------------------------------------------
# Native XML I/O
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "warp"


def _make_warp(n_tilts: int = 5, pixel_size_a: float = 2.0) -> WarpAlignment:
    entries = [
        WarpAlignmentEntry(
            z_index=i,
            tilt_angle=-30.0 + 15.0 * i,
            tilt_axis_angle=85.5 + 0.1 * i,
            tilt_axis_offset_x=20.0 + 1.0 * i,
            tilt_axis_offset_y=-6.0 + 0.4 * i,
        )
        for i in range(n_tilts)
    ]
    return WarpAlignment(
        n_tilts=n_tilts,
        pixel_size_a=pixel_size_a,
        image_dimensions_physical=[1024.0, 1024.0],
        volume_dimensions_physical=[1024.0, 1024.0, 800.0],
        entries=entries,
    )


def _assert_warp_equal(a: WarpAlignment, b: WarpAlignment, rel: float = 1e-8):
    assert a.n_tilts == b.n_tilts
    assert a.pixel_size_a == pytest.approx(b.pixel_size_a, rel=rel)
    assert a.image_dimensions_physical == pytest.approx(b.image_dimensions_physical, rel=rel)
    assert a.volume_dimensions_physical == pytest.approx(b.volume_dimensions_physical, rel=rel)
    for ea, eb in zip(a.entries, b.entries):
        assert ea.z_index == eb.z_index
        assert ea.tilt_angle == pytest.approx(eb.tilt_angle, rel=rel)
        assert ea.tilt_axis_angle == pytest.approx(eb.tilt_axis_angle, rel=rel)
        assert ea.tilt_axis_offset_x == pytest.approx(eb.tilt_axis_offset_x, rel=rel)
        assert ea.tilt_axis_offset_y == pytest.approx(eb.tilt_axis_offset_y, rel=rel)


def test_xml_file_roundtrip(tmp_path):
    """to_file → from_file preserves every field; the PixelSize stamp makes the
    read side work without an explicit pixel size."""
    warp = _make_warp()
    xml_path = tmp_path / "TS_1.xml"
    warp.to_file(xml_path)

    content = xml_path.read_text()
    assert "ImageDimensionsAngstrom" in content
    assert "VolumeDimensionsAngstrom" in content

    warp_rt = WarpAlignment.from_file(xml_path)
    _assert_warp_equal(warp, warp_rt)


def test_xml_string_roundtrip():
    """from_string(str(warp)) round-trips without any kwargs."""
    warp = _make_warp(n_tilts=3, pixel_size_a=1.7005)
    warp_rt = WarpAlignment.from_string(str(warp))
    _assert_warp_equal(warp, warp_rt)


def test_e2e_aln_to_warp_file_and_back(tmp_path):
    """.aln → Alignment → to_warp → XML file → from_warp_file → to_aretomo
    preserves the global alignment fields through the on-disk format."""
    aln = _make_simple_aln(n_tilts=5, pixel_size_a=2.0)
    pixel_size_a = 2.0

    alignment = Alignment.from_aretomo3(aln, vol_size=(512, 512, 400))
    warp = alignment.to_warp(pixel_size_a=pixel_size_a, image_size_px=(512, 512))
    xml_path = tmp_path / "TS_e2e.xml"
    warp.to_file(xml_path)

    alignment_rt = Alignment.from_warp_file(
        xml_path,
        vol_size=(512, 512, 400),
        pixel_size_a=pixel_size_a,
    )
    aln_rt = alignment_rt.to_aretomo(ts_size=(512, 512, 5))

    assert len(aln_rt.GlobalAlignments) == len(aln.GlobalAlignments)
    for orig, rt in zip(aln.GlobalAlignments, aln_rt.GlobalAlignments):
        assert orig.sec == rt.sec
        assert orig.tilt == pytest.approx(rt.tilt, abs=1e-4)
        assert orig.rot == pytest.approx(rt.rot, abs=1e-4)
        assert orig.tx == pytest.approx(rt.tx, abs=1e-4)
        assert orig.ty == pytest.approx(rt.ty, abs=1e-4)


def test_missing_pixel_size_raises():
    """An XML without a CTF PixelSize param and no explicit value must refuse
    loudly instead of guessing (a silent 0 would zero all shifts downstream)."""
    xml = "<TiltSeries><Angles>-30\n0\n30</Angles></TiltSeries>"
    with pytest.raises(ValueError, match="pixel_size_a"):
        WarpAlignment.from_string(xml)


def test_no_angles_rejected():
    with pytest.raises(ValueError, match="Angles"):
        WarpAlignment.from_string("<TiltSeries></TiltSeries>", pixel_size_a=2.0)


def test_dims_absent_read_as_zeros():
    """Older Warp exports carry no dimension attributes — they read as zeros
    (pinned behavior), and absent per-tilt elements default to zeros."""
    xml = "<TiltSeries><Angles>-30\n0\n30</Angles></TiltSeries>"
    warp = WarpAlignment.from_string(xml, pixel_size_a=2.0)
    assert warp.n_tilts == 3
    assert warp.image_dimensions_physical == [0.0, 0.0]
    assert warp.volume_dimensions_physical == [0.0, 0.0, 0.0]
    assert warp.entries[1].tilt_axis_angle == 0.0
    assert warp.entries[2].tilt_axis_offset_x == 0.0


def test_per_tilt_length_mismatch_rejected():
    xml = "<TiltSeries><Angles>-30\n0\n30</Angles><AxisAngle>85.5\n85.6</AxisAngle></TiltSeries>"
    with pytest.raises(ValueError, match="AxisAngle"):
        WarpAlignment.from_string(xml, pixel_size_a=2.0)


def test_real_warp_export_fixture():
    """A real (warpylib-written, EMPIAR-10499-derived) Warp XML: UTF-8 BOM,
    no dimension attributes, stamped default PixelSize."""
    warp = WarpAlignment.from_file(DATA_DIR / "00254.xml", pixel_size_a=1.7005)
    assert warp.n_tilts == 41
    assert warp.pixel_size_a == 1.7005  # explicit value wins over the stamp
    assert warp.image_dimensions_physical == [0.0, 0.0]
    assert min(e.tilt_angle for e in warp.entries) == pytest.approx(-60.01, abs=1e-6)
    assert all(abs(e.tilt_axis_offset_x) < 1e4 for e in warp.entries)

    # without the explicit value, the stamped CTF PixelSize is used
    warp_stamped = WarpAlignment.from_file(DATA_DIR / "00254.xml")
    assert warp_stamped.pixel_size_a == 1.0


def test_native_io_cross_validated_against_warpylib(tmp_path):
    """Local-only: field-by-field comparison of the native I/O against
    warpylib's TiltSeries loader/writer. Skips wherever warpylib (or a
    binary-compatible torch) is unavailable — including CI."""
    try:
        import torch
        from warpylib import TiltSeries
    except Exception:
        pytest.skip("warpylib not importable")

    warp = _make_warp()
    xml_path = tmp_path / "native.xml"
    warp.to_file(xml_path)

    # native write → warpylib read
    ts = TiltSeries(path=str(xml_path))
    assert int(ts.n_tilts) == warp.n_tilts
    for i, e in enumerate(warp.entries):
        assert float(ts.angles[i]) == pytest.approx(e.tilt_angle, abs=1e-4)
        assert float(ts.tilt_axis_angles[i]) == pytest.approx(e.tilt_axis_angle, abs=1e-4)
        assert float(ts.tilt_axis_offset_x[i]) == pytest.approx(e.tilt_axis_offset_x, abs=1e-3)
        assert float(ts.tilt_axis_offset_y[i]) == pytest.approx(e.tilt_axis_offset_y, abs=1e-3)
    assert [float(v) for v in ts.image_dimensions_physical] == pytest.approx(
        warp.image_dimensions_physical,
        abs=1e-3,
    )
    assert [float(v) for v in ts.volume_dimensions_physical] == pytest.approx(
        warp.volume_dimensions_physical,
        abs=1e-3,
    )

    # warpylib write → native read
    ts2 = TiltSeries(n_tilts=warp.n_tilts)
    ts2.angles = torch.tensor([e.tilt_angle for e in warp.entries], dtype=torch.float32)
    ts2.tilt_axis_angles = torch.tensor([e.tilt_axis_angle for e in warp.entries], dtype=torch.float32)
    ts2.tilt_axis_offset_x = torch.tensor([e.tilt_axis_offset_x for e in warp.entries], dtype=torch.float32)
    ts2.tilt_axis_offset_y = torch.tensor([e.tilt_axis_offset_y for e in warp.entries], dtype=torch.float32)
    ts2.image_dimensions_physical = torch.tensor(warp.image_dimensions_physical, dtype=torch.float32)
    ts2.volume_dimensions_physical = torch.tensor(warp.volume_dimensions_physical, dtype=torch.float32)
    wl_path = tmp_path / "warpylib.xml"
    ts2.save_meta(str(wl_path))

    warp_rt = WarpAlignment.from_file(wl_path, pixel_size_a=warp.pixel_size_a)
    _assert_warp_equal(warp, warp_rt, rel=1e-4)
