"""G2 / G12: the Warp side of the codec against arewarpion's warpylib-backed ``WarpTiltSeriesModel``.

A rigid XML written by ``WarpAlignment`` is patched with (a) a single-node ``GridMovementX = 7 Å``,
(b) a constant ``GridVolumeWarp = (3, -2, 5) Å`` and (c) a ``1×1×T`` per-tilt ``GridMovementY`` — all of
which change every projection although no grid "varies spatially". ``WarpAlignment`` must fold them
(``is_rigid``), ``Alignment.from_warp`` must carry them into the hub, and the CETS chain must reproduce
``WarpTiltSeriesModel.project_volume`` (grids retained) with non-zero ``LevelAngleX``/``LevelAngleY`` and
a ``UseTilt=False`` row. warpylib evaluates in float32, hence the 1e-2 Å tolerance.
"""

import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cets_data_model")
warp_xml = pytest.importorskip("arewarpion.io.warp_xml")
warp_ts = pytest.importorskip("arewarpion.models.warp_ts")

from cryoet_alignment.io.cets.alignment import ReferenceVolume, alignment_to_cets, fold_projection  # noqa: E402
from cryoet_alignment.io.cets.entities import tilt_series_entity, tomogram_entity  # noqa: E402
from cryoet_alignment.io.cets.frames import FRAME_CONVENTIONS, image_frame  # noqa: E402
from cryoet_alignment.io.cryoet_data_portal import Alignment  # noqa: E402
from cryoet_alignment.io.cryoet_data_portal.alignment import PerSectionAlignmentParameters, ang2mat  # noqa: E402
from cryoet_alignment.io.warp import WarpAlignment  # noqa: E402


def _hub(n_rows: int, dark: int, xrot: float) -> Alignment:
    rng = np.random.default_rng(11)
    params = []
    for z in range(n_rows):
        if z == dark:
            continue
        params.append(
            PerSectionAlignmentParameters(
                z_index=z,
                tilt_angle=-42.0 + 6.0 * z + 0.02,
                volume_x_rotation=xrot,
                in_plane_rotation=ang2mat(84.1 + 0.05 * z).tolist(),
                x_offset=float(rng.normal(0, 20)),
                y_offset=float(rng.normal(0, 20)),
            ),
        )
    return Alignment(
        affine_transformation_matrix=np.eye(4).tolist(), alignment_type="GLOBAL", format="WARP",
        is_portal_standard=True, tilt_offset=0.0, volume_offset={"x": 0, "y": 0, "z": 0}, x_rotation_offset=0.0,
        per_section_alignment_parameters=params, volume_dimension={"x": 4095 * 1.7, "y": 4096 * 1.7, "z": 1201 * 1.7},
    )


def _patch_grids(xml: str, n: int, movement_y: list) -> str:
    xml = re.sub(r'(<GridMovementX[^>]*>\s*<Node X="0" Y="0" Z="0" Value=")0(")', r"\g<1>7\g<2>", xml)
    for name, v in (("GridVolumeWarpX", 3.0), ("GridVolumeWarpY", -2.0), ("GridVolumeWarpZ", 5.0)):
        xml = re.sub(rf'(<{name}[^>]*>\s*<Node X="0" Y="0" Z="0" W="0" Value=")0(")', rf"\g<1>{v}\g<2>", xml)
    nodes = "".join(f'<Node X="0" Y="0" Z="{t}" Value="{movement_y[t]}" />' for t in range(n))
    xml = re.sub(
        r"<GridMovementY[^>]*>.*?</GridMovementY>",
        f'<GridMovementY Width="1" Height="1" Depth="{n}" MarginX="0" MarginY="0" MarginZ="0">{nodes}</GridMovementY>',
        xml,
        flags=re.S,
    )
    return xml


@pytest.mark.parametrize("xrot,level_y", [(0.0, 0.0), (0.7, 1.5)])
def test_g2_constant_grids_are_rigid_and_folded(tmp_path: Path, xrot, level_y):
    s = 1.7
    n_rows, dark = 8, 3
    hub = _hub(n_rows, dark, xrot)
    warp = hub.to_warp(pixel_size_a=s, image_size_px=(4095, 4096), n_rows=n_rows, dark_angles={dark: 0.0})
    # LevelAngleY is a gauge: move it out of the angles so the XML exercises the non-zero attribute
    warp.level_angle_y = level_y
    for e in warp.entries:
        e.tilt_angle -= level_y
    movement_y = [0.5 * t - 1.0 for t in range(n_rows)]
    xml = _patch_grids(str(warp), n_rows, movement_y)
    path = tmp_path / "TS.xml"
    path.write_text(xml)

    w = WarpAlignment.from_file(path, pixel_size_a=s)
    assert w.is_rigid and not w.grid_audit.has_angle_grids
    assert w.grid_audit.constant_volume_warp == [3.0, -2.0, 5.0]
    assert [e.movement_x for e in w.entries] == [7.0] * n_rows
    assert [e.movement_y for e in w.entries] == pytest.approx(movement_y)
    assert w.level_angle_x == pytest.approx(xrot) and w.level_angle_y == pytest.approx(level_y)
    assert [e.use_tilt for e in w.entries] == [z != dark for z in range(n_rows)]

    # torch reference with the grids retained
    series = warp_xml.load_warp_tiltseries(path)
    model = warp_ts.WarpTiltSeriesModel(series.ts)
    vol = np.array([hub.volume_dimension[k] for k in "xyz"])
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.15, 0.85, size=(200, 3)) * vol
    ref_xy, valid = model.project_volume(torch.tensor(pts, dtype=torch.float64))
    ref_xy = ref_xy.numpy()

    hub2 = Alignment.from_warp(w)
    assert [p.z_index for p in hub2.per_section_alignment_parameters] == [z for z in range(n_rows) if z != dark]
    assert all(abs(p.volume_x_rotation - xrot) < 1e-9 for p in hub2.per_section_alignment_parameters)
    for p, q in zip(hub.per_section_alignment_parameters, hub2.per_section_alignment_parameters):
        assert abs(p.tilt_angle - q.tilt_angle) < 1e-9

    vol_px = (4095, 4096, 1201)
    ts = tilt_series_entity(tilt_series_id="TS", path=None, width=4095, height=4096, pixel_size_a=s,
                            nominal_angles=[0.0] * n_rows)
    tomo = tomogram_entity(tomogram_id="TS_tomo", path=None, size_px=vol_px, voxel_size_a=s, tilt_series_id="TS")
    ref = ReferenceVolume.from_tomogram(tomo)
    img = image_frame(ts.images[0])
    cets = alignment_to_cets(hub2, tilt_series_id="TS", alignment_name="warp", image=img, reference=ref,
                             frame=FRAME_CONVENTIONS["WARP"])
    p_c = pts - ref.cets_centre_a()
    kept = [z for z in range(n_rows) if z != dark]
    worst = 0.0
    for pa, z in zip(cets.projection_alignments, kept):
        r, t = fold_projection(pa)
        q = (p_c @ r.T)[:, :2] + t + np.array(img.origin_index) * s
        worst = max(worst, float(np.abs(q - ref_xy[z]).max()))
    assert worst < 1e-2, f"CETS chain vs WarpTiltSeriesModel: {worst:.3e} Å"

    # negative control: ignoring the constant grids is wrong by ~7 Å
    w_naive = w.model_copy(deep=True)
    for e in w_naive.entries:
        e.movement_x = e.movement_y = 0.0
    w_naive.grid_audit.constant_volume_warp = [0.0, 0.0, 0.0]
    hub_naive = Alignment.from_warp(w_naive)
    cets_naive = alignment_to_cets(hub_naive, tilt_series_id="TS", alignment_name="warp", image=img, reference=ref,
                                   frame=FRAME_CONVENTIONS["WARP"])
    naive_worst = 0.0
    for pa, z in zip(cets_naive.projection_alignments, kept):
        r, t = fold_projection(pa)
        q = (p_c @ r.T)[:, :2] + t + np.array(img.origin_index) * s
        naive_worst = max(naive_worst, float(np.abs(q - ref_xy[z]).max()))
    assert naive_worst > 5.0


def test_g3_x_rotation_representability():
    hub = _hub(5, -1, 0.0)
    hub.per_section_alignment_parameters[2].volume_x_rotation = 0.2
    with pytest.raises(ValueError, match="volume_x_rotation varies"):
        hub.to_warp(pixel_size_a=2.0, image_size_px=(512, 512))
    for p in hub.per_section_alignment_parameters:
        p.volume_x_rotation = 0.2
    w = hub.to_warp(pixel_size_a=2.0, image_size_px=(512, 512))
    assert w.level_angle_x == pytest.approx(0.2)
    with pytest.raises(ValueError, match="X rotation"):
        hub.to_aretomo(ts_size=(512, 512, 5))
