"""Annotations of the cets-rigid/0.2 profile: the Euler port, the entity round trip and transform folding, the
three particle-star flavours (synthetic and real excerpts), the M import golden, the companion section."""

from pathlib import Path

import numpy as np
import pytest
import starfile

m = pytest.importorskip("cets_data_model.models.models")

from cryoet_alignment.io.cets.annotations import (  # noqa: E402
    annotation_id,
    annotation_kind,
    annotation_points,
    fold_annotation_transform,
    mask_entity,
    point_set_entity,
    select_annotations,
    tomogram_frame,
)
from cryoet_alignment.io.cets.companion import AnnotationCompanion, Companion  # noqa: E402
from cryoet_alignment.io.cets.entities import (  # noqa: E402
    dataset_entity,
    dump_json,
    load_dataset,
    region_entity,
    tomogram_entity,
    validate_document,
)
from cryoet_alignment.io.cets.euler import matrices_to_zyz, matrix_to_zyz, zyz_to_matrices, zyz_to_matrix  # noqa: E402
from cryoet_alignment.io.cets.particles_star import (  # noqa: E402
    StarRows,
    detect_flavour,
    read_particle_star,
    series_stem,
    validate_series_names,
    write_particle_star,
)
from cryoet_alignment.io.cets.profile import ANNOTATION_TO_TOMOGRAM, PHYSICAL_CS  # noqa: E402
from cryoet_alignment.io.relion.alignment import rot_y, rot_z  # noqa: E402

DATA = Path(__file__).parent / "data" / "particles"


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    e = np.column_stack([rng.uniform(-180, 180, n), rng.uniform(0, 180, n), rng.uniform(-180, 180, n)])
    return zyz_to_matrices(e), e


# ------------------------------------------------------------------ Euler port


def test_euler_port_matches_relion_composition_and_roundtrips():
    mats, e = _random_rotations(500)
    for (rot, tilt, psi), a in zip(e, mats):
        # RELION's A(rot, tilt, psi) is the transpose of the active Rz(rot)·Ry(tilt)·Rz(psi)
        assert np.allclose(a, (rot_z(rot) @ rot_y(tilt) @ rot_z(psi)).T, atol=1e-12)
        r2 = matrix_to_zyz(a)
        assert np.abs(zyz_to_matrix(*r2) - a).max() < 1e-12
    scipy = pytest.importorskip("scipy.spatial.transform")
    for (rot, tilt, psi), a in zip(e[:50], mats[:50]):
        # what the cryoET Data Portal stores for RELION-sourced oriented points (point_converter.py:350)
        portal = scipy.Rotation.from_euler("ZYZ", (rot, tilt, psi), degrees=True).inv().as_matrix()
        assert np.allclose(portal, a, atol=1e-12)


def test_euler_gimbal_lock_branches():
    for tilt in (0.0, 180.0):
        a = zyz_to_matrix(20.0, tilt, 50.0)
        rot, t2, psi = matrix_to_zyz(a)
        assert rot == 0.0 and abs(t2 - tilt) < 1e-9
        assert np.abs(zyz_to_matrix(rot, t2, psi) - a).max() < 1e-9
    with pytest.raises(ValueError, match="orthonormal"):
        matrix_to_zyz(np.eye(3) * 1.1)


# ------------------------------------------------------------------ entities, frames, folding


def _scene(size=(1261, 1259, 367), voxel=4.99):
    tomo = tomogram_entity(tomogram_id="TS_1_tomo", path=None, size_px=size, voxel_size_a=voxel, tilt_series_id="TS_1")
    return tomo


@pytest.mark.parametrize("size", [(1260, 1260, 368), (1261, 1259, 367)])
def test_point_entities_roundtrip_and_frames(tmp_path, size):
    tomo = _scene(size)
    fr = tomogram_frame(tomo)
    assert np.allclose(fr.corner_a, [float(n // 2) * 4.99 for n in size])
    assert np.allclose(fr.float_centre_delta_a, [(n / 2 - n // 2) * 4.99 for n in size])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1500, 1500, (25, 3))
    mats, _ = _random_rotations(25, seed=2)
    oriented = point_set_entity(
        annotation_id=annotation_id(tomo.id, "picks"),
        tomogram_id=tomo.id,
        points_a=pts,
        matrices=mats,
    )
    plain = point_set_entity(annotation_id=annotation_id(tomo.id, "plain"), tomogram_id=tomo.id, points_a=pts)
    mask = mask_entity(
        annotation_id=annotation_id(tomo.id, "mask"),
        tomogram_id=tomo.id,
        path="x.zarr",
        size_px=size,
        voxel_size_a=4.99,
    )
    region = region_entity(region_id="TS_1", tomograms=[tomo], annotations=[oriented, plain, mask])
    ds = dataset_entity("d", [region])
    validate_document(ds)
    path = dump_json(ds, tmp_path / "d.cets.json")
    back = load_dataset(path).regions[0]
    assert [annotation_kind(a) for a in back.annotations] == ["oriented_points", "points", "mask"]
    r = annotation_points(back.annotations[0], back)
    assert np.allclose(r.points_a, pts, atol=1e-12) and np.allclose(r.matrices, mats, atol=1e-12)
    assert np.allclose(r.points_corner_a, pts + fr.corner_a)
    r2 = annotation_points(back.annotations[1], back)
    assert r2.matrices is None and np.allclose(r2.points_a, pts)
    # the mask has its own array_to_physical (same grid here) and the identity annotation_to_tomogram
    mfr = tomogram_frame(back.annotations[2])
    assert mfr.size_px == tuple(size) and abs(mfr.spacing_a - 4.99) < 1e-12
    a, t = fold_annotation_transform(back.annotations[2])
    assert np.allclose(a, np.eye(3)) and np.allclose(t, 0)
    # selection helpers
    assert [x.id for x in select_annotations(back, kinds=["mask"])] == [mask.id]
    assert len(select_annotations(back, tomogram_id=tomo.id)) == 3
    with pytest.raises(ValueError, match="not in region"):
        select_annotations(back, ids=["nope"])


def test_annotation_to_tomogram_chains_fold_and_others_are_refused():
    tomo = _scene()
    pts = np.array([[10.0, 20.0, 30.0], [-5.0, 0.0, 2.5]])
    ann = point_set_entity(
        annotation_id="a",
        tomogram_id=tomo.id,
        points_a=pts,
        matrices=zyz_to_matrices([[10, 20, 30], [0, 90, 0]]),
    )
    rot = rot_z(30.0)
    ann.coordinate_transformations = [
        m.Sequence(
            name=ANNOTATION_TO_TOMOGRAM,
            input=PHYSICAL_CS,
            output=PHYSICAL_CS,
            sequence=[
                m.Translation(name="shift", translation=[1.0, 2.0, 3.0]),
                m.Affine(name="rot", affine=rot.tolist()),
                m.Scale(name="s", scale=[2.0, 2.0, 2.0]),
            ],
        ),
    ]
    region = region_entity(region_id="r", tomograms=[tomo], annotations=[ann])
    r = annotation_points(ann, region)
    expect = 2.0 * (rot @ (pts + [1.0, 2.0, 3.0]).T).T
    assert np.allclose(r.points_a, expect, atol=1e-12)
    assert np.allclose(r.matrices[0], rot @ zyz_to_matrix(10, 20, 30), atol=1e-12)
    ann.coordinate_transformations = [m.Identity(name="something_else", input=PHYSICAL_CS, output=PHYSICAL_CS)]
    with pytest.raises(ValueError, match="annotation_to_tomogram"):
        annotation_points(ann, region)
    ann.coordinate_transformations = [
        m.MapAxis(
            name=ANNOTATION_TO_TOMOGRAM,
            input=PHYSICAL_CS,
            output=PHYSICAL_CS,
            map_axis=[m.AxisNameMapping(axis1_name=n, axis2_name=n) for n in "xyz"],
        ),
    ]
    with pytest.raises(ValueError, match="unsupported step"):
        annotation_points(ann, region)
    ann.coordinate_transformations = [m.Identity(name=ANNOTATION_TO_TOMOGRAM, input=PHYSICAL_CS, output=PHYSICAL_CS)]
    ann.source_tomogram_id = "missing"
    with pytest.raises(ValueError, match="missing"):
        annotation_points(ann, region)
    with pytest.raises(ValueError, match="det"):
        point_set_entity(annotation_id="b", tomogram_id=tomo.id, points_a=pts, matrices=[np.eye(3), -np.eye(3)])


# ------------------------------------------------------------------ star flavours


def test_series_stem_and_flavour_detection():
    assert series_stem("../tomostar/TS_01.tomostar") == "TS_01"
    assert series_stem("C:\\data\\TS_01.tomostar") == "TS_01"
    assert series_stem("TS_01.mrc") == "TS_01" and series_stem("TS_01") == "TS_01"
    import pandas as pd

    assert (
        detect_flavour(
            pd.DataFrame(
                {
                    "rlnCenteredCoordinateXAngst": [0],
                    "rlnCenteredCoordinateYAngst": [0],
                    "rlnCenteredCoordinateZAngst": [0],
                },
            ),
            None,
        )
        == "relion5"
    )
    assert detect_flavour(pd.DataFrame({"rlnCoordinateX": [0], "rlnMicrographName": ["a"]}), None) == "warp"
    assert detect_flavour(pd.DataFrame({"rlnCoordinateX": [0], "rlnTomoName": ["a"]}), None) == "warp"
    assert (
        detect_flavour(
            pd.DataFrame({"rlnCoordinateX": [0], "rlnTomoName": ["a"]}),
            pd.DataFrame({"rlnOpticsGroup": [1], "rlnTomoTiltSeriesPixelSize": [1.5]}),
        )
        == "relion5"
    )


@pytest.mark.parametrize("flavour", ["warp", "m", "relion5"])
def test_star_write_read_identity(tmp_path, flavour):
    rng = np.random.default_rng(3)
    n = 20
    pos = rng.uniform(0, 6000, (n, 3))
    mats, _ = _random_rotations(n, seed=4)
    series = ["TS_01"] * 10 + ["TS_02"] * 10
    rows = StarRows(series=series, positions_corner_a=pos, matrices=mats, extra={"rlnRandomSubset": [1, 2] * 10})
    extent = np.array([6307.84, 6307.84, 3080.0])
    p = write_particle_star(
        tmp_path / f"{flavour}.star",
        flavour,
        rows,
        coords_angpix=4.99,
        extent_a=extent,
        voltage_kv=300.0,
    )
    # auto-detection: warp and m share their column set (only the pixel-size chain differs); relion5 is distinct
    auto = read_particle_star(p, "auto", coords_angpix=4.99 if flavour == "warp" else None)
    assert auto.flavour == {"warp": "warp", "m": "warp", "relion5": "relion5"}[flavour]
    t = read_particle_star(p, flavour, coords_angpix=4.99 if flavour == "warp" else None)
    assert t.flavour == flavour and t.series == series
    got = t.positions_corner_a + (extent / 2.0 if t.centred_input else 0.0)
    assert np.abs(got - pos).max() < 1e-5  # star precision %.6f on px / Å
    assert np.abs(np.einsum("nij,nkj->nik", t.matrices, mats) - np.eye(3)).max() < 1e-7
    assert t.extra["rlnRandomSubset"] == [1, 2] * 10
    assert t.coords_angpix == pytest.approx(4.99)
    assert validate_series_names(t, ["TS_01"]) == ["TS_02.tomostar"]
    raw = starfile.read(p, always_dict=True)
    blk = raw["particles"]
    if flavour == "relion5":
        assert {"rlnTomoName", "rlnCenteredCoordinateXAngst", "rlnTomoSubtomogramRot"} <= set(blk.columns)
        assert raw["optics"]["rlnTomoTiltSeriesPixelSize"].iloc[0] == pytest.approx(4.99)
    else:
        assert {"rlnMicrographName", "rlnCoordinateX", "rlnAngleRot"} <= set(blk.columns)
        assert blk["rlnMicrographName"].iloc[0] == "TS_01.tomostar"
        if flavour == "m":
            assert raw["optics"]["rlnImagePixelSize"].iloc[0] == pytest.approx(4.99)
            assert "rlnOriginXAngst" in blk.columns
        else:
            assert "optics" not in raw and "rlnOriginX" in blk.columns


def test_warp_star_origin_variants_and_explicit_pixel(tmp_path):
    import pandas as pd

    coords = np.array([[100.0, 200.0, 50.0], [10.5, 20.25, 30.125]])
    # (a) rlnOriginX in px, no pixel columns -> --coords-angpix required
    df = pd.DataFrame(
        {
            "rlnMicrographName": ["TS_01.tomostar", "TS_02.tomostar"],
            "rlnCoordinateX": coords[:, 0],
            "rlnCoordinateY": coords[:, 1],
            "rlnCoordinateZ": coords[:, 2],
            "rlnOriginX": [1.0, -2.0],
            "rlnOriginY": [0.0, 0.5],
            "rlnOriginZ": [0.25, 0.0],
        },
    )
    starfile.write(df, tmp_path / "a.star")
    with pytest.raises(ValueError, match="coords-angpix"):
        read_particle_star(tmp_path / "a.star", "warp")
    t = read_particle_star(tmp_path / "a.star", "warp", coords_angpix=2.0)
    assert np.allclose(t.positions_corner_a, (coords - [[1.0, 0.0, 0.25], [-2.0, 0.5, 0.0]]) * 2.0)
    assert t.matrices is None
    # M reading the same px origins with a distinct shift pixel
    t = read_particle_star(tmp_path / "a.star", "m", coords_angpix=2.0, angpix_shifts=4.0)
    assert np.allclose(t.positions_corner_a, coords * 2.0 - np.array([[1.0, 0.0, 0.25], [-2.0, 0.5, 0.0]]) * 4.0)
    # (b) rlnOriginXAngst with rlnImagePixelSize per row (Warp divides per row, then scales by coords_angpix)
    df2 = df.drop(columns=["rlnOriginX", "rlnOriginY", "rlnOriginZ"]).assign(
        rlnOriginXAngst=[3.0, 6.0],
        rlnOriginYAngst=[0.0, 0.0],
        rlnOriginZAngst=[1.5, 0.0],
        rlnImagePixelSize=[1.5, 3.0],
    )
    starfile.write(df2, tmp_path / "b.star")
    t = read_particle_star(tmp_path / "b.star", "warp", coords_angpix=2.0)
    assert np.allclose(t.positions_corner_a, (coords - [[2.0, 0.0, 1.0], [2.0, 0.0, 0.0]]) * 2.0)
    with pytest.raises(ValueError, match="varies"):
        read_particle_star(tmp_path / "b.star", "warp")  # rlnImagePixelSize differs between rows


def test_m_import_golden_matches_m_species_table():
    """The RELION star fed to ``MTools create_species`` (M chain: optics rlnImagePixelSize 4.0, not the tilt-series
    pixel 1.54) reproduces the species' own particle table before refinement."""
    t = read_particle_star(DATA / "m_import_run_data.star", "m")
    assert t.coords_angpix == pytest.approx(4.0) and t.coords_angpix_source == "rlnImagePixelSize"
    mt = starfile.read(DATA / "m_species_particles.star")
    m_xyz = mt[["wrpCoordinateX1", "wrpCoordinateY1", "wrpCoordinateZ1"]].to_numpy(float)
    assert [s.replace(".tomostar", "") for s in mt["wrpSourceName"]] == t.series
    assert np.abs(m_xyz - t.positions_corner_a).max() < 1e-3
    m_mat = zyz_to_matrices(mt[["wrpAngleRot1", "wrpAngleTilt1", "wrpAnglePsi1"]].to_numpy(float))
    assert np.abs(np.einsum("nij,nkj->nik", m_mat, t.matrices) - np.eye(3)).max() < 1e-6
    # the same star read as RELION 5 legacy uses the tilt-series pixel: a different (RELION's) interpretation
    t5 = read_particle_star(DATA / "m_import_run_data.star", "auto")
    assert t5.flavour == "relion5" and t5.coords_angpix == pytest.approx(1.54)
    assert t5.extra["rlnRandomSubset"] == t.extra["rlnRandomSubset"]


def test_relion5_legacy_and_centred_decode():
    """Legacy px rows: a direct port of ParticleSet::getPosition (coord·ts_pix − A_sub·origin) on the Warp 2D export
    excerpt; centred rows: write/read identity including the subtomogram rotation of the origin."""
    t = read_particle_star(DATA / "warp_2d_export.star", "auto")
    raw = starfile.read(DATA / "warp_2d_export.star", always_dict=True)
    p = raw["particles"]
    ts_pix = float(raw["optics"]["rlnTomoTiltSeriesPixelSize"].iloc[0])
    direct = p[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float) * ts_pix - p[
        ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    ].to_numpy(float)
    assert t.flavour == "relion5" and not t.centred_input
    assert np.abs(t.positions_corner_a - direct).max() < 1e-9
    assert t.extra["rlnTomoParticleId"] == p["rlnTomoParticleId"].tolist()


def test_relion5_subtomogram_matrix_and_centred_coordinates(tmp_path):
    import pandas as pd

    sub = np.array([[10.0, 20.0, 30.0], [-40.0, 100.0, 5.0]])
    part = np.array([[3.0, 80.0, -20.0], [15.0, 45.0, 60.0]])
    origin = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])
    xyz = np.array([[100.0, -50.0, 20.0], [-300.0, 12.5, -7.0]])
    df = pd.DataFrame(
        {
            "rlnTomoName": ["TS_01.tomostar", "TS_01.tomostar"],
            "rlnCenteredCoordinateXAngst": xyz[:, 0],
            "rlnCenteredCoordinateYAngst": xyz[:, 1],
            "rlnCenteredCoordinateZAngst": xyz[:, 2],
            "rlnOriginXAngst": origin[:, 0],
            "rlnOriginYAngst": origin[:, 1],
            "rlnOriginZAngst": origin[:, 2],
            "rlnAngleRot": part[:, 0],
            "rlnAngleTilt": part[:, 1],
            "rlnAnglePsi": part[:, 2],
            "rlnTomoSubtomogramRot": sub[:, 0],
            "rlnTomoSubtomogramTilt": sub[:, 1],
            "rlnTomoSubtomogramPsi": sub[:, 2],
            "rlnOpticsGroup": [1, 1],
        },
    )
    starfile.write(
        {"optics": pd.DataFrame({"rlnOpticsGroup": [1], "rlnTomoTiltSeriesPixelSize": [1.54]}), "particles": df},
        tmp_path / "c.star",
    )
    t = read_particle_star(tmp_path / "c.star")
    a_sub = zyz_to_matrices(sub)
    expect = xyz - np.einsum("nij,nj->ni", a_sub, origin)
    assert t.centred_input and np.abs(t.positions_corner_a - expect).max() < 1e-9
    assert np.abs(t.matrices - np.einsum("nij,njk->nik", a_sub, zyz_to_matrices(part))).max() < 1e-12
    # export with an odd extent and read back: the float centre is honoured exactly
    extent = np.array([4095 * 1.54, 4096 * 1.54, 1001 * 1.54])
    rows = StarRows(series=["TS_01", "TS_01"], positions_corner_a=expect + extent / 2, matrices=t.matrices)
    p2 = write_particle_star(tmp_path / "d.star", "relion5", rows, coords_angpix=1.54, extent_a=extent)
    t2 = read_particle_star(p2)
    assert np.abs(t2.positions_corner_a - expect).max() < 1e-5
    e2 = matrices_to_zyz(t2.matrices)
    assert np.abs(zyz_to_matrices(e2) - t.matrices).max() < 1e-7


# ------------------------------------------------------------------ companion


def test_companion_annotations_section_and_old_schema_loads(tmp_path):
    c = Companion(generator="t")
    c.annotations["a"] = AnnotationCompanion(
        kind="oriented_points",
        tomogram_id="TS_1_tomo",
        flavour="warp",
        coords_angpix_a=4.99,
        columns={"rlnRandomSubset": [1, 2]},
        metadata={"annotation_ingest_id": "x-1"},
    )
    p = c.dump(tmp_path / "x.cets-companion.json")
    back = Companion.load(p)
    assert (
        back.annotations["a"].columns == {"rlnRandomSubset": [1, 2]}
        and back.annotations["a"].metadata["annotation_ingest_id"] == "x-1"
    )
    old = p.read_text().replace("cets-rigid-companion/0.2", "cets-rigid-companion/0.1")
    (tmp_path / "old.cets-companion.json").write_text(old)
    assert Companion.load(tmp_path / "old.cets-companion.json").annotations["a"].kind == "oriented_points"
    with pytest.raises(ValueError, match="companion schema"):
        (tmp_path / "bad.json").write_text(old.replace("0.1", "9.9"))
        Companion.load(tmp_path / "bad.json")
