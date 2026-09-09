"""Numerical gates of the cets-rigid/0.1 codec against arewarpion's pinned torch projection models.

Skipped when arewarpion (torch) is not importable. G1: odd image/volume sizes — the CETS chain evaluated
purely from the document must reproduce ``AretomoTsModel.project_volume_global`` to 1e-9 px, and the naive
"copy the native shifts" recipe must NOT (negative control, ~1.6 px on odd sizes). G8: the real
24jul16a ``.aln`` and the RELION matrix model.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
arewarpion = pytest.importorskip("arewarpion")

from arewarpion.models.aretomo_ts import AretomoTsModel  # noqa: E402
from cryoet_alignment.io.aretomo3 import AreTomo3ALN  # noqa: E402
from cryoet_alignment.io.aretomo3.aln import GlobalAlignmentInfo  # noqa: E402
from cryoet_alignment.io.cets.alignment import (  # noqa: E402
    ReferenceVolume,
    alignment_from_cets,
    alignment_to_cets,
    fold_projection,
)
from cryoet_alignment.io.cets.entities import tilt_series_entity, tomogram_entity  # noqa: E402
from cryoet_alignment.io.cets.frames import FRAME_CONVENTIONS, ImageFrame, image_frame  # noqa: E402
from cryoet_alignment.io.cryoet_data_portal import Alignment  # noqa: E402

pytest.importorskip("cets_data_model")

TESTDATA = Path("/hpc/projects/group.czii/utz.ermel/repos/arewarpo/testdata_runs/at3_24jul16a")


def _synthetic_aln(n_tilts: int, n_raw: int) -> AreTomo3ALN:
    rng = np.random.default_rng(1)
    rows = []
    secs = np.sort(rng.choice(np.arange(1, n_raw + 1), size=n_tilts, replace=False))
    for i, sec in enumerate(secs):
        rows.append(
            GlobalAlignmentInfo(
                sec=int(sec),
                rot=-96.3 + 0.01 * i,
                tx=float(rng.normal(0, 30)),
                ty=float(rng.normal(0, 30)),
                tilt=-45.0 + 3.0 * i + 0.03,
            ),
        )
    from cryoet_alignment.io.aretomo3.aln import DarkFrameInfo

    darks = [
        DarkFrameInfo(section_idx=z, val2=z + 1, angle=0.0) for z in range(n_raw) if (z + 1) not in set(secs.tolist())
    ]
    return AreTomo3ALN(RawSize=(0, 0, n_raw), NumPatches=0, DarkFrames=darks, GlobalAlignments=rows)


def _cets_project(cets_alignment, points_a_corner, ref: ReferenceVolume, img: ImageFrame):
    """Evaluate the document chain on canonical corner-origin points, returning corner-origin image Å."""
    p_c = np.asarray(points_a_corner) - ref.cets_centre_a()
    out = []
    for pa in cets_alignment.projection_alignments:
        r, t = fold_projection(pa)
        q_c = (p_c @ r.T)[:, :2] + t
        out.append(q_c + np.array(img.origin_index) * np.array(img.spacing_a))
    return np.stack(out)  # (T, N, 2)


@pytest.mark.parametrize("image_n,vol", [((4096, 4096), (4096, 4096, 1196)), ((4095, 4097), (4095, 4097, 2001))])
def test_g1_aretomo3_chain_matches_torch_model(image_n, vol):
    s = 1.54
    n_raw, n_tilts = 9, 7
    aln = _synthetic_aln(n_tilts, n_raw)
    aln = aln.model_copy(update={"RawSize": (image_n[0], image_n[1], n_raw)})
    hub = Alignment.from_aretomo3(aln, vol_size_px=vol, pixel_size_a=s)

    model = AretomoTsModel(
        rot_deg=torch.tensor([g.rot for g in aln.GlobalAlignments], dtype=torch.float64),
        tilt_deg=torch.tensor([g.tilt for g in aln.GlobalAlignments], dtype=torch.float64),
        shifts_px=torch.tensor([[g.tx, g.ty] for g in aln.GlobalAlignments], dtype=torch.float64),
        raw_size_px=image_n,
        pixel_size_a=s,
        volume_dims_a=tuple(v * s for v in vol),
    )
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.1, 0.9, size=(200, 3)) * np.array(vol) * s  # canonical corner-origin Å
    ref_xy, _ = model.project_volume_global(torch.tensor(pts, dtype=torch.float64))
    ref_xy = ref_xy.numpy()

    ts = tilt_series_entity(
        tilt_series_id="TS",
        path=None,
        width=image_n[0],
        height=image_n[1],
        pixel_size_a=s,
        nominal_angles=[0.0] * n_raw,
    )
    tomo = tomogram_entity(tomogram_id="TS_tomo", path=None, size_px=vol, voxel_size_a=s, tilt_series_id="TS")
    ref = ReferenceVolume.from_tomogram(tomo)
    img = image_frame(ts.images[0])

    cets = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="aretomo3", image=img, reference=ref)
    got = _cets_project(cets, pts, ref, img)
    err_px = np.abs(got - ref_xy).max() / s
    assert err_px < 1e-9, f"CETS chain vs AretomoTsModel: {err_px:.3e} px"

    # negative control: the naive recipe (native shifts copied, no centre deltas)
    naive = cets.model_copy(deep=True)
    for pa, p in zip(
        naive.projection_alignments,
        sorted(hub.per_section_alignment_parameters, key=lambda q: q.z_index),
    ):
        pa.sequence[2].translation = [p.x_offset * s, p.y_offset * s]
    naive_err = np.abs(_cets_project(naive, pts, ref, img) - ref_xy).max() / s
    odd = any(n % 2 for n in image_n) or any(n % 2 for n in vol)
    if odd:
        assert naive_err > 0.5, f"negative control should fail on odd sizes, got {naive_err:.3e} px"
    else:
        assert naive_err < 1e-9

    # and back: the hub is reproduced exactly (offsets in px)
    back = alignment_from_cets(
        cets,
        tilt_series=ts,
        reference=ref,
        target_frame=FRAME_CONVENTIONS["ARETOMO3"],
        native_dimension_a=hub.volume_dimension,
        format_="ARETOMO3",
    )
    for a, b in zip(hub.per_section_alignment_parameters, back.per_section_alignment_parameters):
        assert a.z_index == b.z_index
        assert abs(a.x_offset - b.x_offset) < 1e-9 and abs(a.y_offset - b.y_offset) < 1e-9
        assert abs(a.tilt_angle - b.tilt_angle) < 1e-9 and abs(a.tilt_axis_rotation - b.tilt_axis_rotation) < 1e-9
        assert abs(b.volume_x_rotation) < 1e-9


@pytest.mark.skipif(not TESTDATA.exists(), reason="real AreTomo3 run not available")
def test_g8_real_aln_chain_matches_torch_model():
    aln_path = sorted((TESTDATA / "outB").glob("*.aln"))[0]
    aln = AreTomo3ALN.from_file(aln_path)
    s = 1.54
    vol = (aln.RawSize[0], aln.RawSize[1], 1196)
    hub = Alignment.from_aretomo3(aln, vol_size_px=vol, pixel_size_a=s)
    model = AretomoTsModel(
        rot_deg=torch.tensor([g.rot for g in aln.GlobalAlignments], dtype=torch.float64),
        tilt_deg=torch.tensor([g.tilt for g in aln.GlobalAlignments], dtype=torch.float64),
        shifts_px=torch.tensor([[g.tx, g.ty] for g in aln.GlobalAlignments], dtype=torch.float64),
        raw_size_px=(aln.RawSize[0], aln.RawSize[1]),
        pixel_size_a=s,
        volume_dims_a=tuple(v * s for v in vol),
    )
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.1, 0.9, size=(200, 3)) * np.array(vol) * s
    ref_xy, _ = model.project_volume_global(torch.tensor(pts, dtype=torch.float64))
    ts = tilt_series_entity(
        tilt_series_id="TS",
        path=None,
        width=vol[0],
        height=vol[1],
        pixel_size_a=s,
        nominal_angles=[0.0] * aln.n_raw,
    )
    tomo = tomogram_entity(
        tomogram_id="TS_tomo",
        path=None,
        size_px=(1024, 1024, 299),
        voxel_size_a=vol[0] * s / 1024,
        tilt_series_id="TS",
    )
    ref = ReferenceVolume.from_tomogram(tomo)
    img = image_frame(ts.images[0])
    cets = alignment_to_cets(hub, tilt_series_id="TS", alignment_name="aretomo3", image=img, reference=ref)
    got = _cets_project(cets, pts, ref, img)
    err_px = np.abs(got - ref_xy.numpy()).max() / s
    assert err_px < 1e-3, f"{err_px:.3e} px"
