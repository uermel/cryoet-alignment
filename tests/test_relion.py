"""Tests for the RELION 5 tilt-series alignment format.

Covers the RelionAlignment model (two-file RELION-5 layout write, both-layout
read, matrix-vs-Euler precedence, the projection-matrix decomposition ported
from RELION's Tomogram::getProjectionAnglesFromMatrix), the conversion math in
Alignment.from_relion / to_relion, and a real WarpTools-written fixture.
A final cross-validation against arewarpion's independent RELION star reader
runs only where that package happens to be installed.
"""

from pathlib import Path

import numpy as np
import pytest
from cryoet_alignment import read, write
from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.aretomo3.aln import GlobalAlignmentInfo
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.relion import RelionAlignment, RelionAlignmentEntry
from cryoet_alignment.io.relion.alignment import eulers_from_matrix, projection_matrix

DATA_DIR = Path(__file__).parent / "data" / "relion"
REAL_STAR = DATA_DIR / "matching_tomograms.star"
REAL_NAMES = [
    "24jul16a_Position_27_1.tomostar",
    "24jul16a_Position_18_1.tomostar",
    "24jul16a_Position_16_3.tomostar",
]
MATRIX_LABELS = ("rlnTomoProjX", "rlnTomoProjY", "rlnTomoProjZ", "rlnTomoProjW")


def _make_relion(n_tilts: int = 5, pixel_size_a: float = 2.0) -> RelionAlignment:
    entries = [
        RelionAlignmentEntry(
            z_index=i,
            nominal_stage_tilt_angle=-30.0 + 15.0 * i,
            x_tilt=0.5,
            y_tilt=-30.0 + 15.0 * i + 0.2,
            z_rot=85.5 + 0.1 * i,
            x_shift_angst=20.0 + 1.0 * i,
            y_shift_angst=-6.0 + 0.4 * i,
            pre_exposure=3.0 * i,
        )
        for i in range(n_tilts)
    ]
    return RelionAlignment(
        tomo_name="TS_1",
        pixel_size_a=pixel_size_a,
        volume_size_px=(512, 512, 400),
        hand=-1.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        entries=entries,
    )


def _make_simple_aln(n_tilts: int = 5) -> AreTomo3ALN:
    global_alignments = [
        GlobalAlignmentInfo(
            sec=i + 1,
            rot=85.5 + 0.1 * i,
            gmag=1.0,
            tx=10.0 + 0.5 * i,
            ty=-3.0 + 0.2 * i,
            smean=1.0,
            sfit=1.0,
            scale=1.0,
            base=0.0,
            tilt=-30.0 + 15.0 * i,
        )
        for i in range(n_tilts)
    ]
    return AreTomo3ALN(
        RawSize=(512, 512, n_tilts),
        NumPatches=0,
        DarkFrames=[],
        AlphaOffset=0.0,
        BetaOffset=0.0,
        GlobalAlignments=global_alignments,
    )


def _assert_relion_equal(a: RelionAlignment, b: RelionAlignment, abs_tol: float = 1e-5):
    assert a.tomo_name == b.tomo_name
    assert a.pixel_size_a == pytest.approx(b.pixel_size_a, abs=abs_tol)
    assert a.volume_size_px == b.volume_size_px
    assert a.hand == pytest.approx(b.hand, abs=abs_tol)
    assert a.voltage == pytest.approx(b.voltage, abs=abs_tol)
    assert len(a.entries) == len(b.entries)
    for ea, eb in zip(a.entries, b.entries):
        assert ea.z_index == eb.z_index
        for f in (
            "nominal_stage_tilt_angle",
            "x_tilt",
            "y_tilt",
            "z_rot",
            "x_shift_angst",
            "y_shift_angst",
            "pre_exposure",
        ):
            assert getattr(ea, f) == pytest.approx(getattr(eb, f), abs=abs_tol)


# ---------------------------------------------------------------------------
# Format I/O
# ---------------------------------------------------------------------------


def test_two_file_roundtrip(tmp_path):
    """RELION-5 layout write -> read preserves every field (starfile %.6f)."""
    relion = _make_relion()
    star = tmp_path / "tomograms.star"
    relion.to_file(star)

    assert star.exists()
    ts_star = tmp_path / "tilt_series" / "TS_1.star"
    assert ts_star.exists(), "per-tomogram star not written to tilt_series/"
    assert star.read_text().startswith("# version 50001")
    assert "rlnTomoTiltSeriesStarFile" in star.read_text()

    relion_rt = RelionAlignment.from_file(star)
    _assert_relion_equal(relion, relion_rt)


def test_write_refuses_missing_metadata(tmp_path):
    relion = _make_relion()
    relion.voltage = None
    with pytest.raises(ValueError, match="voltage"):
        relion.to_file(tmp_path / "tomograms.star")


def test_multi_tomogram_selection():
    with pytest.raises(ValueError, match="pass tomo_name"):
        RelionAlignment.from_file(REAL_STAR)
    with pytest.raises(ValueError, match="not in"):
        RelionAlignment.from_file(REAL_STAR, tomo_name="nope", image_size_px=(4096, 4096))


def test_embedded_layout_euler_read(tmp_path):
    """relion-4-style single file with embedded per-tomogram block, Euler columns."""
    star = tmp_path / "embedded.star"
    star.write_text(
        """
data_global

loop_
_rlnTomoName #1
_rlnTomoTiltSeriesPixelSize #2
_rlnTomoSizeX #3
_rlnTomoSizeY #4
_rlnTomoSizeZ #5
_rlnTomoHand #6
TS_E 2.0 512 512 400 -1.0

data_TS_E

loop_
_rlnTomoNominalStageTiltAngle #1
_rlnTomoXTilt #2
_rlnTomoYTilt #3
_rlnTomoZRot #4
_rlnTomoXShiftAngst #5
_rlnTomoYShiftAngst #6
-30.0 0.5 -29.8 85.5 20.0 -6.0
0.0 0.5 0.2 85.6 21.0 -5.6
30.0 0.5 30.2 85.7 22.0 -5.2
""",
    )
    relion = read(star, tomo_name="TS_E")  # .star inference -> reader="relion"
    assert isinstance(relion, RelionAlignment)
    assert relion.n_tilts == 3
    assert relion.voltage is None  # optional metadata absent
    assert relion.entries[1].y_tilt == 0.2
    assert relion.entries[2].z_rot == 85.7
    assert relion.entries[0].x_tilt == 0.5
    assert relion.entries[0].pre_exposure == 0.0  # column absent -> 0


def test_missing_mandatory_columns(tmp_path):
    star = tmp_path / "bad.star"
    star.write_text(
        """
data_global

loop_
_rlnTomoName #1
_rlnTomoTiltSeriesPixelSize #2
_rlnTomoSizeX #3
_rlnTomoSizeY #4
_rlnTomoSizeZ #5
TS_B 2.0 512 512 400

data_TS_B

loop_
_rlnTomoYTilt #1
-30.0
""",
    )
    with pytest.raises(ValueError, match="rlnTomoZRot"):
        RelionAlignment.from_file(star)


# ---------------------------------------------------------------------------
# Projection-matrix decomposition (tomogram.cpp:66-117 port)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dims", [((512, 512, 400), (512, 512)), ((511, 513, 401), (511, 513))])
def test_matrix_decomposition_golden(dims):
    """P built by the setProjectionMatrix transcription -> Eulers/shifts recovered,
    for even AND odd dimensions.

    For odd volume dimensions the recovered shifts differ from the input by
    EXACTLY pix*[R @ (N/2 - N//2)]_xy — RELION's own internal asymmetry
    (setProjectionMatrix centers on N//2, getProjectionAnglesFromMatrix on N/2),
    reproduced faithfully by the port. Angles are always exact."""
    vol, img = dims
    pix = 2.0
    for xt, yt, zr, sx, sy in [
        (0.0, -30.0, 85.5, 20.0, -6.0),
        (1.5, 45.0, -95.0, -13.7, 8.2),
        (-2.0, 0.0, 0.0, 0.0, 0.0),
    ]:
        p = projection_matrix(xt, yt, zr, sx, sy, vol, img, pix)
        rxt, ryt, rzr, rsx, rsy = eulers_from_matrix(p, vol, img, pix)
        assert (rxt, ryt, rzr) == pytest.approx((xt, yt, zr), abs=1e-9)
        dc = np.array([d / 2.0 - d // 2 for d in vol])
        delta = pix * (p[:3, :3] @ dc)
        assert (rsx, rsy) == pytest.approx((sx + delta[0], sy + delta[1]), abs=1e-6)


def test_matrix_decomposition_gimbal():
    """ytilt = +-90 hits the gimbal branches; the recomposed matrix must match
    even though the individual angles are degenerate."""
    vol, img = (512, 512, 400), (512, 512)
    for yt in (90.0, -90.0):
        p = projection_matrix(0.0, yt, 30.0, 5.0, -3.0, vol, img, 2.0)
        vals = eulers_from_matrix(p, vol, img, 2.0)
        p_re = projection_matrix(*vals, vol, img, 2.0)
        assert np.abs(p_re - p).max() == pytest.approx(0.0, abs=1e-9)


def test_real_warptools_fixture():
    """Genuine WarpTools-written tomograms.star: embedded layout, matrix-only
    per-tilt tables (no Euler columns), rlnTomoHand = -1."""
    relion = RelionAlignment.from_file(REAL_STAR, tomo_name=REAL_NAMES[2], image_size_px=(4096, 4096))
    assert relion.n_tilts == 31
    assert relion.hand == -1.0
    assert relion.pixel_size_a == 1.54
    assert relion.volume_size_px == (4096, 4096, 2000)
    assert relion.voltage == 300.0
    tilts = sorted(e.y_tilt for e in relion.entries)
    assert tilts[0] == pytest.approx(-45.0, abs=0.1)
    assert tilts[-1] == pytest.approx(45.0, abs=0.1)
    assert all(abs(e.x_tilt) < 1.0 for e in relion.entries)

    # every tomogram is selectable
    for name in REAL_NAMES:
        r = RelionAlignment.from_file(REAL_STAR, tomo_name=name, image_size_px=(4096, 4096))
        assert r.tomo_name == name and r.n_tilts == 31


def test_real_matrices_reproduced_from_recovered_eulers():
    """Recomposing P from the recovered Eulers/shifts reproduces the file's
    matrices in every projection-relevant entry (xy rows + rotation). The z
    translation is not representable in Euler+shift form and never enters a
    projection; Warp's centered-convention matrices carry 0 there."""
    import starfile
    from cryoet_alignment.io.relion.alignment import _parse_vector

    relion = RelionAlignment.from_file(REAL_STAR, tomo_name=REAL_NAMES[2], image_size_px=(4096, 4096))
    tilt = starfile.read(REAL_STAR, always_dict=True)[REAL_NAMES[2]]
    for i, e in enumerate(relion.entries):
        p_orig = np.array([_parse_vector(tilt[c].iloc[i]) for c in MATRIX_LABELS])
        p_re = projection_matrix(
            e.x_tilt,
            e.y_tilt,
            e.z_rot,
            e.x_shift_angst,
            e.y_shift_angst,
            relion.volume_size_px,
            (4096, 4096),
            relion.pixel_size_a,
        )
        assert np.abs(p_re[:2] - p_orig[:2]).max() < 1e-3  # xy rows incl. shifts
        assert np.abs(p_re[2, :3] - p_orig[2, :3]).max() < 1e-5  # rotation


def test_matrix_only_requires_image_size():
    with pytest.raises(ValueError, match="image_size_px"):
        RelionAlignment.from_file(REAL_STAR, tomo_name=REAL_NAMES[0])


def test_matrix_precedence_and_fallback(tmp_path):
    """When both matrices and Eulers are present: matrices win when
    image_size_px is given (RELION's own precedence); without it the Euler
    columns are used with a warning."""
    vol, img, pix = (512, 512, 400), (512, 512), 2.0
    p = projection_matrix(0.0, -30.0, 85.5, 20.0, -6.0, vol, img, pix)
    rows = "".join(f"[{','.join(f'{v:.9g}' for v in p[i])}] " for i in range(4))
    star = tmp_path / "both.star"
    star.write_text(
        f"""
data_global

loop_
_rlnTomoName #1
_rlnTomoTiltSeriesPixelSize #2
_rlnTomoSizeX #3
_rlnTomoSizeY #4
_rlnTomoSizeZ #5
TS_P {pix} {vol[0]} {vol[1]} {vol[2]}

data_TS_P

loop_
_rlnTomoYTilt #1
_rlnTomoZRot #2
_rlnTomoXShiftAngst #3
_rlnTomoYShiftAngst #4
_rlnTomoProjX #5
_rlnTomoProjY #6
_rlnTomoProjZ #7
_rlnTomoProjW #8
99.0 99.0 99.0 99.0 {rows}
""",
    )
    # matrices win (deliberately disagreeing Euler columns are ignored)
    relion = RelionAlignment.from_file(star, image_size_px=img)
    assert relion.entries[0].y_tilt == pytest.approx(-30.0, abs=1e-6)
    assert relion.entries[0].x_shift_angst == pytest.approx(20.0, abs=1e-5)

    # no image_size_px -> Euler fallback with a warning
    with pytest.warns(UserWarning, match="image_size_px"):
        relion2 = RelionAlignment.from_file(star)
    assert relion2.entries[0].y_tilt == 99.0


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def test_aretomo3_to_relion_file_and_back(tmp_path):
    """.aln -> Alignment -> to_relion -> two-file star -> from_relion_file ->
    to_aretomo preserves the global alignment fields (RELION's own importer
    mapping: zrot=ROT, ytilt=TILT, shifts = pix*(TX,TY))."""
    aln = _make_simple_aln()
    pix = 2.0

    alignment = Alignment.from_aretomo3(aln, vol_size=(512, 512, 400))
    relion = alignment.to_relion("TS_A", pixel_size_a=pix)
    assert relion.n_tilts == 5
    for ali, e in zip(aln.GlobalAlignments, relion.entries):
        assert e.y_tilt == pytest.approx(ali.tilt, abs=1e-6)
        assert e.z_rot == pytest.approx(ali.rot, abs=1e-6)
        assert e.x_shift_angst == pytest.approx(ali.tx * pix, abs=1e-6)
        assert e.y_shift_angst == pytest.approx(ali.ty * pix, abs=1e-6)
        assert e.x_tilt == 0.0

    star = tmp_path / "tomograms.star"
    relion.to_file(star)
    alignment_rt = Alignment.from_relion_file(star)
    aln_rt = alignment_rt.to_aretomo(ts_size=(512, 512, 5))

    for orig, rt in zip(aln.GlobalAlignments, aln_rt.GlobalAlignments):
        assert orig.sec == rt.sec
        assert orig.tilt == pytest.approx(rt.tilt, abs=1e-4)
        assert orig.rot == pytest.approx(rt.rot, abs=1e-4)
        assert orig.tx == pytest.approx(rt.tx, abs=1e-4)
        assert orig.ty == pytest.approx(rt.ty, abs=1e-4)


def test_relion_to_alignment_to_relion_roundtrip():
    """RELION -> canonical -> RELION preserves all per-tilt fields, including
    the x-tilt, which no other format in this package can carry."""
    relion = _make_relion()
    alignment = Alignment.from_relion(relion)
    assert alignment.format == "RELION"
    assert alignment.per_section_alignment_parameters[0].volume_x_rotation == 0.5
    assert alignment.per_section_alignment_parameters[0].x_offset == pytest.approx(20.0 / 2.0)

    relion_rt = alignment.to_relion(
        "TS_1",
        pixel_size_a=relion.pixel_size_a,
        hand=relion.hand,
        pre_exposures=[e.pre_exposure for e in relion.entries],
    )
    for orig, rt in zip(relion.entries, relion_rt.entries):
        assert rt.x_tilt == pytest.approx(orig.x_tilt, abs=1e-9)
        assert rt.y_tilt == pytest.approx(orig.y_tilt, abs=1e-9)
        assert rt.z_rot == pytest.approx(orig.z_rot, abs=1e-6)
        assert rt.x_shift_angst == pytest.approx(orig.x_shift_angst, abs=1e-6)
        assert rt.y_shift_angst == pytest.approx(orig.y_shift_angst, abs=1e-6)
        assert rt.pre_exposure == pytest.approx(orig.pre_exposure)
        # the nominal stage angle collapses onto the refined tilt (documented)
        assert rt.nominal_stage_tilt_angle == pytest.approx(orig.y_tilt, abs=1e-9)


def test_relion_to_warp_cross_conversion():
    relion = _make_relion()
    alignment = Alignment.from_relion(relion)
    warp = alignment.to_warp(pixel_size_a=relion.pixel_size_a, image_size_px=(512, 512))
    assert warp.n_tilts == relion.n_tilts
    # Warp cannot carry the x-tilt; shifts survive px<->Angstrom exactly
    assert warp.entries[0].tilt_axis_offset_x == pytest.approx(relion.entries[0].x_shift_angst)


# ---------------------------------------------------------------------------
# API registration
# ---------------------------------------------------------------------------


def test_api_read_write(tmp_path):
    relion = _make_relion()
    star = tmp_path / "tomograms.star"
    write(relion, star)  # type inference -> writer="relion"
    relion_rt = read(star, reader="relion")
    _assert_relion_equal(relion, relion_rt)

    inferred = read(star)  # .star -> reader="relion"
    assert isinstance(inferred, RelionAlignment)


def test_api_read_cdp_regression(tmp_path):
    """reader="cdp" used to call a nonexistent Alignment.from_cdp."""
    alignment = Alignment.from_aretomo3(_make_simple_aln(), vol_size=(512, 512, 400))
    out = tmp_path / "alignment.json"
    write(alignment, out)
    back = read(out, reader="cdp")
    assert isinstance(back, Alignment)
    assert len(back.per_section_alignment_parameters) == 5


# ---------------------------------------------------------------------------
# Cross-validation against arewarpion (local-only)
# ---------------------------------------------------------------------------


def test_cross_validated_against_arewarpion(tmp_path):
    """Our written two-file layout read back by arewarpion's independent
    RELION star reader. Skips wherever arewarpion is not installed."""
    try:
        from arewarpion.io.relion_star import read_tomograms_star
    except Exception:
        pytest.skip("arewarpion not importable")

    relion = _make_relion()
    star = tmp_path / "tomograms.star"
    relion.to_file(star)

    data = read_tomograms_star(star)["TS_1"]
    assert data.pixel_size_a == pytest.approx(relion.pixel_size_a)
    assert int(data.hand) == -1
    for i, e in enumerate(relion.entries):
        assert float(data.xtilt_deg[i]) == pytest.approx(e.x_tilt, abs=1e-5)
        assert float(data.ytilt_deg[i]) == pytest.approx(e.y_tilt, abs=1e-5)
        assert float(data.zrot_deg[i]) == pytest.approx(e.z_rot, abs=1e-5)
        assert float(data.xshift_a[i]) == pytest.approx(e.x_shift_angst, abs=1e-4)
        assert float(data.yshift_a[i]) == pytest.approx(e.y_shift_angst, abs=1e-4)
        assert float(data.pre_exposure[i]) == pytest.approx(e.pre_exposure, abs=1e-5)
