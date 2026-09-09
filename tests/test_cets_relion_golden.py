"""For a RELION-origin hub (``floor`` centres) the CETS chain equals ``RelionTomogramModel`` exactly —
no centre deltas, odd sizes included — and the decomposed Eulers are RELION's."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cets_data_model")
relion_ts = pytest.importorskip("arewarpion.models.relion_ts")

from cryoet_alignment.io.cets.alignment import ReferenceVolume, alignment_to_cets, fold_projection  # noqa: E402
from cryoet_alignment.io.cets.entities import tilt_series_entity, tomogram_entity  # noqa: E402
from cryoet_alignment.io.cets.frames import FRAME_CONVENTIONS, image_frame  # noqa: E402
from cryoet_alignment.io.cets.rotation import decompose  # noqa: E402
from cryoet_alignment.io.cryoet_data_portal import Alignment  # noqa: E402
from cryoet_alignment.io.relion import RelionAlignment, RelionAlignmentEntry  # noqa: E402


@pytest.mark.parametrize("image_n,vol", [((4096, 4096), (4096, 4096, 1200)), ((4095, 4097), (4095, 4097, 1201))])
def test_relion_chain_matches_matrix_model(image_n, vol):
    s = 1.9
    rng = np.random.default_rng(2)
    entries = [
        RelionAlignmentEntry(
            z_index=i,
            nominal_stage_tilt_angle=-30.0 + 10 * i,
            x_tilt=0.4,
            y_tilt=-30.0 + 10 * i + 0.1,
            z_rot=-95.0 + 0.02 * i,
            x_shift_angst=float(rng.normal(0, 40)),
            y_shift_angst=float(rng.normal(0, 40)),
            pre_exposure=3.0 * i,
        )
        for i in range(7)
    ]
    rel = RelionAlignment(
        tomo_name="TS",
        pixel_size_a=s,
        volume_size_px=vol,
        hand=-1,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        entries=entries,
    )
    hub = Alignment.from_relion(rel)
    assert hub.volume_dimension == {"x": vol[0] * s, "y": vol[1] * s, "z": vol[2] * s}

    model = relion_ts.RelionTomogramModel(
        xtilt_deg=torch.tensor([e.x_tilt for e in entries], dtype=torch.float64),
        ytilt_deg=torch.tensor([e.y_tilt for e in entries], dtype=torch.float64),
        zrot_deg=torch.tensor([e.z_rot for e in entries], dtype=torch.float64),
        xshift_a=torch.tensor([e.x_shift_angst for e in entries], dtype=torch.float64),
        yshift_a=torch.tensor([e.y_shift_angst for e in entries], dtype=torch.float64),
        tomo_dims_px=vol,
        image_dims_px=image_n,
        pixel_size_a=s,
    )
    pts = rng.uniform(0.1, 0.9, size=(200, 3)) * np.array(vol) * s
    ref_xy, _ = model.project_volume(torch.tensor(pts, dtype=torch.float64))
    ref_xy = ref_xy.numpy()

    ts = tilt_series_entity(
        tilt_series_id="TS",
        path=None,
        width=image_n[0],
        height=image_n[1],
        pixel_size_a=s,
        nominal_angles=[e.nominal_stage_tilt_angle for e in entries],
    )
    tomo = tomogram_entity(tomogram_id="TS_tomo", path=None, size_px=vol, voxel_size_a=s, tilt_series_id="TS")
    ref = ReferenceVolume.from_tomogram(tomo)
    img = image_frame(ts.images[0])
    cets = alignment_to_cets(
        hub,
        tilt_series_id="TS",
        alignment_name="relion",
        image=img,
        reference=ref,
        frame=FRAME_CONVENTIONS["RELION"],
    )
    p_c = pts - ref.cets_centre_a()
    worst = 0.0
    for i, pa in enumerate(cets.projection_alignments):
        r, t = fold_projection(pa)
        # for RELION no deltas exist: the CETS shift IS rlnTomoX/YShiftAngst
        assert t == pytest.approx([entries[i].x_shift_angst, entries[i].y_shift_angst], abs=1e-9)
        rot, tilt, xrot = decompose(r)
        assert (rot, tilt, xrot) == pytest.approx((entries[i].z_rot, entries[i].y_tilt, entries[i].x_tilt), abs=1e-9)
        q = (p_c @ r.T)[:, :2] + t + np.array(img.origin_index) * s
        worst = max(worst, float(np.abs(q - ref_xy[i]).max()))
    assert worst < 1e-9 * s * 4096, f"{worst:.3e} Å"
