"""Unit tests of the cets-rigid/0.2 codec that need no torch: round-trip identity, layouts and folding, frames,
ids and selection, CTF nulls and units, the cets-imod adapter, the companion manifest, the config resolver."""

import json
from pathlib import Path

import numpy as np
import pytest

m = pytest.importorskip("cets_data_model.models.models")

from cryoet_alignment.io.cets import PROFILE_VERSION  # noqa: E402
from cryoet_alignment.io.cets.adapters.cets_imod import alignment_from_cets_imod, is_cets_imod_alignment  # noqa: E402
from cryoet_alignment.io.cets.alignment import (  # noqa: E402
    ReferenceVolume,
    alignment_from_cets,
    alignment_name_of,
    alignment_to_cets,
    fold_projection,
    select_alignment,
    select_tomogram,
)
from cryoet_alignment.io.cets.companion import (  # noqa: E402
    AlignmentCompanion,
    Companion,
    ImageCompanion,
    TiltSeriesCompanion,
)
from cryoet_alignment.io.cets.config import ConfigError, ConfigFile, Resolver  # noqa: E402
from cryoet_alignment.io.cets.ctf import ctf_metadata, from_aretomo3_row, from_warp_values, to_warp_values  # noqa: E402
from cryoet_alignment.io.cets.entities import (  # noqa: E402
    dataset_entity,
    dump_json,
    load_dataset,
    region_entity,
    tilt_series_entity,
    to_json_dict,
    tomogram_entity,
    validate_document,
)
from cryoet_alignment.io.cets.frames import FRAME_CONVENTIONS, array_to_physical, image_frame  # noqa: E402
from cryoet_alignment.io.cets.rotation import decompose, rotation  # noqa: E402
from cryoet_alignment.io.cryoet_data_portal import Alignment  # noqa: E402
from cryoet_alignment.io.cryoet_data_portal.alignment import PerSectionAlignmentParameters, ang2mat  # noqa: E402
from cryoet_alignment.io.imod import ImodAlignment, ImodTLT, ImodXF  # noqa: E402

DATA = Path(__file__).parent / "data"


def _hub(n=6, fmt="ARETOMO3", xrot=0.0, dark=()):
    rng = np.random.default_rng(0)
    params = [
        PerSectionAlignmentParameters(
            z_index=z,
            tilt_angle=-45 + 15 * z + 0.01,
            volume_x_rotation=xrot,
            in_plane_rotation=ang2mat(-96.33 + 0.01 * z).tolist(),
            x_offset=float(rng.normal(0, 50)),
            y_offset=float(rng.normal(0, 50)),
        )
        for z in range(n)
        if z not in dark
    ]
    return Alignment(
        affine_transformation_matrix=np.eye(4).tolist(),
        alignment_type="GLOBAL",
        format=fmt,
        is_portal_standard=True,
        tilt_offset=0.0,
        volume_offset={"x": 0, "y": 0, "z": 0},
        x_rotation_offset=0.0,
        per_section_alignment_parameters=params,
        volume_dimension={"x": 4096 * 1.54, "y": 4096 * 1.54, "z": 1196 * 1.54},
    )


def _scene(n=6, size=(4096, 4096), vol=(1024, 1024, 299), dark=()):
    s = 1.54
    ts = tilt_series_entity(
        tilt_series_id="TS",
        path="TS.mrc",
        width=size[0],
        height=size[1],
        pixel_size_a=s,
        nominal_angles=[-45 + 15 * z for z in range(n)],
    )
    tomo = tomogram_entity(
        tomogram_id="TS_tomo",
        path="TS_Vol.mrc",
        size_px=vol,
        voxel_size_a=4096 * s / vol[0],
        tilt_series_id="TS",
    )
    return ts, tomo, ReferenceVolume.from_tomogram(tomo), image_frame(ts.images[0])


def _assert_hub_equal(a: Alignment, b: Alignment, tol=1e-9):
    assert [p.z_index for p in a.per_section_alignment_parameters] == [
        p.z_index for p in b.per_section_alignment_parameters
    ]
    for p, q in zip(a.per_section_alignment_parameters, b.per_section_alignment_parameters):
        assert abs(p.tilt_angle - q.tilt_angle) < tol
        assert abs(p.tilt_axis_rotation - q.tilt_axis_rotation) < tol
        assert abs(p.volume_x_rotation - q.volume_x_rotation) < tol
        assert abs(p.x_offset - q.x_offset) < tol and abs(p.y_offset - q.y_offset) < tol


# ------------------------------------------------------------------ identity, all frames, odd sizes


@pytest.mark.parametrize("fmt", ["ARETOMO3", "WARP", "RELION"])
@pytest.mark.parametrize("size,vol", [((4096, 4096), (1024, 1024, 299)), ((4095, 4097), (819, 819, 239))])
def test_roundtrip_identity(fmt, size, vol):
    hub = _hub(fmt=fmt, xrot=0.3 if fmt != "ARETOMO3" else 0.0, dark=(2,))
    ts, tomo, ref, img = _scene(size=size, vol=vol)
    cets = alignment_to_cets(hub, tilt_series_id="TS", alignment_name=fmt.lower(), image=img, reference=ref)
    assert [pa.id for pa in cets.projection_alignments][:2] == [
        f"TS_{fmt.lower()}_align_0",
        f"TS_{fmt.lower()}_align_1",
    ]
    assert [pa.tilt_image_id for pa in cets.projection_alignments] == [f"TS_{z}" for z in (0, 1, 3, 4, 5)]
    assert [s.name for s in cets.projection_alignments[0].sequence] == ["tilt", "in_plane_rotation", "shift"]
    back = alignment_from_cets(
        cets,
        tilt_series=ts,
        reference=ref,
        target_frame=FRAME_CONVENTIONS[fmt],
        native_dimension_a=hub.volume_dimension,
        format_=fmt,
    )
    _assert_hub_equal(hub, back)
    validate_document(cets)


def test_frames_matter_for_odd_sizes():
    """The same document decoded into a different frame convention differs by the half-pixel deltas."""
    hub = _hub(fmt="ARETOMO3")
    ts, tomo, ref, img = _scene(size=(4095, 4095), vol=(819, 819, 239))
    cets = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="a", image=img, reference=ref)
    as_relion = alignment_from_cets(
        cets,
        tilt_series=ts,
        reference=ref,
        target_frame=FRAME_CONVENTIONS["RELION"],
        native_dimension_a=hub.volume_dimension,
    )
    d = max(
        abs(p.x_offset - q.x_offset)
        for p, q in zip(hub.per_section_alignment_parameters, as_relion.per_section_alignment_parameters)
    )
    assert 0.1 < d < 2.0


# ------------------------------------------------------------------ generic folding, layouts, rejections


def test_two_entry_and_reordered_layouts_fold_identically():
    hub = _hub()
    ts, tomo, ref, img = _scene()
    cets = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="a", image=img, reference=ref)
    pa = cets.projection_alignments[3]
    r3, t3 = fold_projection(pa)
    # 2-entry [Affine R, Translation]
    two = pa.model_copy(
        update={
            "sequence": [
                m.Affine(
                    name="rotation",
                    affine=(np.array(pa.sequence[1].affine) @ np.array(pa.sequence[0].affine)).tolist(),
                ),
                pa.sequence[2],
            ],
        },
    )
    r2, t2 = fold_projection(two)
    assert np.allclose(r2, r3) and np.allclose(t2, t3)
    # names other than the profile's are folded generically too
    renamed = pa.model_copy(deep=True)
    renamed.sequence[0].name = "whatever"
    assert np.allclose(fold_projection(renamed)[0], r3)
    # rejected: wrong operator name, non-rotation affine, 2D-homogeneous affine with a translation column
    with pytest.raises(ValueError, match="tomogram_to_projection"):
        fold_projection(pa.model_copy(update={"name": "align_projection_image_to_tomogram"}))
    bad = pa.model_copy(deep=True)
    bad.sequence[1].affine = [[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]]
    with pytest.raises(ValueError, match="not orthonormal"):
        fold_projection(bad)
    refl = pa.model_copy(deep=True)
    refl.sequence[0].affine = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="det"):
        fold_projection(refl)


def test_rotation_decompose_roundtrip_and_gimbal():
    for rot, tilt, xrot in [(-96.33, -45.03, 0.0), (10.0, 60.0, -2.5), (170.0, -89.9, 1.0), (33.0, 0.0, 0.0)]:
        assert decompose(rotation(rot, tilt, xrot)) == pytest.approx((rot, tilt, xrot), abs=1e-9)
    r = rotation(20.0, 90.0, 0.0)
    rot, tilt, xrot = decompose(r)
    assert tilt == pytest.approx(90.0) and np.allclose(rotation(rot, tilt, xrot), r)


# ------------------------------------------------------------------ frames


def test_frames_validate_chain_and_bare_scale_is_corner_origin():
    ts, tomo, ref, img = _scene(size=(4095, 4096))
    assert img.origin_index == (2047.0, 2048.0) and img.spacing_a == (1.54, 1.54)
    assert ReferenceVolume.from_tomogram(tomo).cets_centre_a()[0] == pytest.approx(512 * 4096 * 1.54 / 1024)
    im = ts.images[0]
    im.coordinate_transformations = [
        m.Scale(name="array_to_physical", input="array", output="physical", scale=[1.54, 1.54]),
    ]
    fr = image_frame(im)
    assert fr.origin_index == (0.0, 0.0)  # never an implied centering
    im.coordinate_transformations = [m.Scale(name="pixel_size", input="array", output="physical", scale=[1.54])]
    with pytest.raises(ValueError, match="exactly one transform named 'array_to_physical'"):
        image_frame(im)
    im.coordinate_transformations = [array_to_physical((4095, 4096), (1.54, 1.60))]
    with pytest.raises(ValueError, match="anisotropic"):
        image_frame(im).isotropic_spacing  # noqa: B018
    im.coordinate_systems = im.coordinate_systems[:1]
    with pytest.raises(ValueError, match="missing coordinate system 'physical'"):
        image_frame(im)


def test_volume_reconciliation_refuses_mismatch():
    """Portal alignment 18924 reports 1260 × 4.99 = 6287.4 Å although the raw extent is 4096 × 1.54 = 6307.84 Å:
    against a tomogram at its implied voxel that box is 4 voxels short and is refused; the raw box reconciles,
    and so does the 630-voxel tomogram at ITS implied voxel."""
    hub = _hub()
    hub.volume_dimension = {"x": 1260 * 4.99, "y": 1260 * 4.99, "z": 368 * 4.99}
    ts, tomo, ref, img = _scene(vol=(1260, 1260, 368))  # implied voxel 5.00622
    with pytest.raises(ValueError, match="disagree"):
        alignment_to_cets(hub, tilt_series_id="TS", alignment_name="a", image=img, reference=ref)
    raw_box = {"x": 4096 * 1.54, "y": 4096 * 1.54, "z": 368 * 4096 * 1.54 / 1260}
    alignment_to_cets(
        hub,
        tilt_series_id="TS",
        alignment_name="a",
        image=img,
        reference=ref,
        native_dimension_a=raw_box,
    )
    tomo_ok = tomogram_entity(
        tomogram_id="ok",
        path=None,
        size_px=(630, 630, 184),
        voxel_size_a=4096 * 1.54 / 630,
        tilt_series_id="TS",
    )
    alignment_to_cets(
        hub,
        tilt_series_id="TS",
        alignment_name="a",
        image=img,
        reference=ReferenceVolume.from_tomogram(tomo_ok),
        native_dimension_a=raw_box,
    )


# ------------------------------------------------------------------ ids, selection


def test_two_alignments_distinct_ids_and_explicit_selection():
    hub = _hub()
    ts, tomo, ref, img = _scene()
    a1 = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="aretomo3", image=img, reference=ref)
    a2 = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="portal18924", image=img, reference=ref)
    tomo2 = tomogram_entity(
        tomogram_id="TS_tomo2",
        path=None,
        size_px=(630, 630, 184),
        voxel_size_a=4096 * 1.54 / 630,
        tilt_series_id="TS",
    )
    region = region_entity(region_id="TS", tilt_series=[ts], alignments=[a1, a2], tomograms=[tomo, tomo2])
    ids = [pa.id for a in region.alignments for pa in a.projection_alignments]
    assert len(ids) == len(set(ids))
    validate_document(dataset_entity("d", [region]))
    assert alignment_name_of(a2) == "portal18924"
    with pytest.raises(ValueError, match="select one with --alignment"):
        select_alignment(region)
    assert select_alignment(region, "portal18924") is a2
    assert select_alignment(region, 0) is a1
    with pytest.raises(ValueError, match="select one with --tomogram"):
        select_tomogram(region, a1)
    assert select_tomogram(region, a1, "TS_tomo2") is tomo2
    comp = Companion(alignments=[AlignmentCompanion(name="aretomo3", tilt_series_id="TS", tomogram_ids=["TS_tomo"])])
    assert select_tomogram(region, a1, companion=comp) is tomo
    comp = Companion(
        alignments=[
            AlignmentCompanion(
                name="aretomo3",
                tilt_series_id="TS",
                reference_tomogram_id="TS_tomo2",
                tomogram_ids=["TS_tomo", "TS_tomo2"],
            ),
        ],
    )
    assert select_tomogram(region, a1, companion=comp) is tomo2


# ------------------------------------------------------------------ CTF nulls and units


def test_defocus_handedness_explicit_null_survives():
    ctf = ctf_metadata(defocus_u_a=20000, defocus_v_a=21000, defocus_angle_deg=200.0)
    assert (ctf.defocus_u, ctf.defocus_v, ctf.defocus_angle) == (21000.0, 20000.0, 110.0)
    assert ctf.defocus_handedness is None and ctf.phase_shift is None
    ts, tomo, ref, img = _scene(n=1)
    ts.images[0].ctf_metadata = ctf
    d = to_json_dict(ts)
    assert d["images"][0]["ctf_metadata"]["defocus_handedness"] is None
    back = m.TiltSeries.model_validate(d)
    assert back.images[0].ctf_metadata.defocus_handedness is None
    assert m.CTFMetadata(defocus_u=1.0).defocus_handedness == -1  # what the model would invent otherwise
    from cryoet_alignment.io.aretomo3.ctf import CtfInfo

    row = CtfInfo(
        micrograph=1,
        df_max_a=21717.31,
        df_min_a=20740.78,
        azimuth_deg=52.0,
        phase_rad=0.5,
        score=0.1,
        res_a=9.6,
        df_hand=-1,
    )
    c = from_aretomo3_row(row)
    assert c.phase_shift == pytest.approx(np.degrees(0.5)) and c.defocus_handedness is None
    w = from_warp_values(2.2167659, 0.0076084905, 17.0, 0.25)
    assert w.defocus_u == pytest.approx((2.2167659 + 0.0076084905 / 2) * 1e4) and w.phase_shift == 45.0
    assert to_warp_values(w) == pytest.approx((2.2167659, 0.0076084905, 17.0, 0.25))


# ------------------------------------------------------------------ cets-imod adapter


def test_cets_imod_adapter_matches_from_imod():
    xf = ImodXF.from_file(DATA / "convert" / "imod_1" / "tilt_1.xf")
    tlt = ImodTLT.from_file(DATA / "convert" / "imod_1" / "tilt_1.tlt")
    apix = 2.0
    ts = tilt_series_entity(
        tilt_series_id="tilt_1",
        path="tilt_1.mrc",
        width=1024,
        height=1024,
        pixel_size_a=apix,
        nominal_angles=list(tlt.angles),
    )
    pas = []
    for i, x in enumerate(xf.alignments):
        rot = x.rot_matrix()
        pas.append(
            m.ProjectionAlignment(
                id=f"tilt_1_align_{i}",
                tilt_image_id=f"tilt_1_{i}",
                name="IMOD projection alignment from a .xf file.",
                input="Tilt-image",
                output="Aligned tilt-image",
                sequence=[
                    m.Translation(
                        name="IMOD translation from a .xf file. Shifts in angstroms.",
                        translation=[x.sx * apix, x.sy * apix, 0.0],
                    ),
                    m.Affine(
                        name="IMOD rotation from a .xf file.",
                        affine=[[rot[0, 0], rot[0, 1], 0.0], [rot[1, 0], rot[1, 1], 0.0], [0.0, 0.0, 1.0]],
                    ),
                ],
            ),
        )
    doc = m.Alignment(tilt_series_id="tilt_1", projection_alignments=pas)
    assert is_cets_imod_alignment(doc)
    with pytest.raises(ValueError, match="tomogram_to_projection"):
        fold_projection(pas[0])
    got = alignment_from_cets_imod(doc, tilt_series=ts, pixel_size_a=apix)
    exp = Alignment.from_imod(ImodAlignment(xf=xf, tlt=tlt, xtilt=None, tiltcom=None, newstcom=None))
    _assert_hub_equal(exp, got, tol=1e-9)
    ts_odd = tilt_series_entity(
        tilt_series_id="tilt_1",
        path=None,
        width=1023,
        height=1024,
        pixel_size_a=apix,
        nominal_angles=list(tlt.angles),
    )
    with pytest.raises(ValueError, match="even image sizes"):
        alignment_from_cets_imod(doc, tilt_series=ts_odd, pixel_size_a=apix)


# ------------------------------------------------------------------ companion + documents


def test_companion_roundtrip_and_paths(tmp_path):
    comp = Companion(
        generator="test",
        tilt_series={
            "TS": TiltSeriesCompanion(
                voltage_kv=300,
                images={"TS_0": ImageCompanion(acquisition_index_1b=3, exposure_dose=100.0)},
            ),
        },
        alignments=[
            AlignmentCompanion(name="aretomo3", tilt_series_id="TS", tomogram_ids=["TS_tomo"], dropped=["locals"]),
        ],
    )
    doc = tmp_path / "run.cets.json"
    path = Companion.path_for(doc)
    assert path.name == "run.cets-companion.json"
    comp.dump(path)
    assert Companion.load_for(doc).tilt_series["TS"].images["TS_0"].exposure_dose == 100.0
    assert Companion.load_for(tmp_path / "other.cets.json") is None
    bad = json.loads(path.read_text())
    bad["schema_version"] = "x"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="companion schema"):
        Companion.load(tmp_path / "bad.json")


def test_document_dump_load(tmp_path):
    hub = _hub()
    ts, tomo, ref, img = _scene()
    cets = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="a", image=img, reference=ref)
    ds = dataset_entity("d", [region_entity(region_id="TS", tilt_series=[ts], alignments=[cets], tomograms=[tomo])])
    p = dump_json(ds, tmp_path / "d.cets.json")
    back = load_dataset(p)
    assert (
        back.regions[0].alignments[0].projection_alignments[0].sequence[2].translation
        == cets.projection_alignments[0].sequence[2].translation
    )
    assert PROFILE_VERSION == "cets-rigid/0.2"


# ------------------------------------------------------------------ resolver / config


def test_resolver_chain_and_default_warning(tmp_path):
    cfg = tmp_path / "cets.yaml"
    cfg.write_text(
        "cets:\n  voltage: 300\ncets-aretomo3:\n  to-cets: {mdoc_dir: mdoc}\nseries:\n  TS_02: {pix: 1.34}\n",
    )
    conf = ConfigFile.load(cfg, known_options={"voltage", "mdoc_dir", "pix", "cs"})
    msgs = []
    r = Resolver(
        "cets-aretomo3",
        "to-cets",
        cli={"cs": 2.7, "voltage": None},
        config=conf,
        series="TS_02",
        warn=msgs.append,
    )
    assert r.value("cs") == 2.7 and r.report[-1].source == "cli"
    assert r.value("voltage") == 300 and r.report[-1].source == "config"
    assert r.value("mdoc_dir") == "mdoc" and r.report[-1].source == "config"
    assert r.value("pix", discovered=1.54) == 1.34 and r.report[-1].source == "config"  # series beats discovery
    r2 = Resolver("cets-aretomo3", "to-cets", config=conf, series="TS_01", warn=msgs.append)
    assert r2.value("pix", discovered=1.54) == 1.54 and r2.report[-1].source == "discovered"
    assert r2.value("dose_convention", default="exclusive") == "exclusive" and r2.report[-1].source == "default"
    assert msgs == [
        "WARNING: dose_convention defaulted to 'exclusive'; set it with --dose-convention or config key series.TS_01.dose_convention",
    ]
    with pytest.raises(ConfigError, match="--amp-contrast"):
        r2.require("amp_contrast")
    (tmp_path / "bad.yaml").write_text("cets:\n  volt: 3\n")
    with pytest.raises(ConfigError, match="unknown option cets.'volt'"):
        ConfigFile.load(tmp_path / "bad.yaml", known_options={"voltage"})
    assert any("[config]" in line for line in r.lines())
