"""Adapter for documents written by TomoBabel's ``cets-imod`` (``imod/converters/tilt_series.py``).

cets-imod encodes a projection alignment as the IMOD ``.xf`` transform of the tilt image
(``input="Tilt-image"``, ``output="Aligned tilt-image"``): ``sequence=[Translation(dx·apix, dy·apix, 0),
Affine([[a11, a12, 0], [a21, a22, 0], [0, 0, 1]])]`` — a 2D image→aligned-image operator without the tilt
angle, which lives in ``TiltImage.nominal_tilt_angle`` (the refined ``.tlt`` value). It emits no
``array_to_physical`` scale, so the pixel size must be supplied. This adapter maps such a document onto the
hub exactly like ``Alignment.from_imod`` maps the ``.xf``/``.tlt`` pair (``imod2are``), restricted to EVEN
image sizes until IMOD's rotation-centre convention is pinned for odd ones. Anything that does not match
the cets-imod shape is rejected naming the element.
"""

from typing import Optional

import numpy as np

from cryoet_alignment.io.cryoet_data_portal.alignment import Alignment, PerSectionAlignmentParameters, imod2are

CETS_IMOD_INPUT = "Tilt-image"
CETS_IMOD_OUTPUT = "Aligned tilt-image"


def _ttype(t) -> str:
    v = getattr(t, "transformation_type", None)
    return str(getattr(v, "value", v) or "")


def is_cets_imod_alignment(cets_alignment) -> bool:
    pas = cets_alignment.projection_alignments or []
    return bool(pas) and all(pa.input == CETS_IMOD_INPUT and pa.output == CETS_IMOD_OUTPUT for pa in pas)


def alignment_from_cets_imod(
    cets_alignment,
    *,
    tilt_series,
    pixel_size_a: float,
    volume_dimension_a: Optional[dict] = None,
) -> Alignment:
    """cets-imod document -> hub ``Alignment`` (format ``IMOD``)."""
    images = {im.id: im for im in (tilt_series.images or [])}
    width = height = None
    params = []
    for pa in cets_alignment.projection_alignments or []:
        if pa.input != CETS_IMOD_INPUT or pa.output != CETS_IMOD_OUTPUT:
            raise ValueError(f"projection alignment {pa.id!r}: not a cets-imod alignment ({pa.input!r} -> {pa.output!r})")
        steps = list(pa.sequence or [])
        if len(steps) != 2 or _ttype(steps[0]) != "translation" or _ttype(steps[1]) != "affine":
            raise ValueError(f"projection alignment {pa.id!r}: cets-imod expects [Translation, Affine], got {[_ttype(s) for s in steps]}")
        im = images.get(pa.tilt_image_id)
        if im is None or im.section is None or im.nominal_tilt_angle is None:
            raise ValueError(f"projection alignment {pa.id!r}: tilt image {pa.tilt_image_id!r} missing section/nominal_tilt_angle")
        width, height = im.width, im.height
        if width is None or height is None or width % 2 or height % 2:
            raise ValueError(
                f"tilt image {im.id!r}: cets-imod documents are only accepted for even image sizes "
                f"(got {width}x{height}); IMOD's centre convention for odd sizes is not pinned",
            )
        a = np.array(steps[1].affine, dtype=np.float64)
        if a.shape != (3, 3) or not np.allclose(a[2], [0, 0, 1]) or not np.allclose(a[:2, 2], 0):
            raise ValueError(f"projection alignment {pa.id!r}: affine is not a cets-imod 2D rotation block")
        tr = np.array(steps[0].translation, dtype=np.float64)
        shift_px = tr[:2] / pixel_size_a
        mat, shift = imod2are(a[:2, :2], shift_px)
        params.append(
            PerSectionAlignmentParameters(
                z_index=int(im.section),
                tilt_angle=float(im.nominal_tilt_angle),
                volume_x_rotation=0.0,
                in_plane_rotation=mat.tolist(),
                x_offset=float(shift[0]),
                y_offset=float(shift[1]),
            ),
        )
    params.sort(key=lambda p: p.z_index)
    return Alignment(
        affine_transformation_matrix=np.eye(4).tolist(),
        alignment_type="GLOBAL",
        format="IMOD",
        is_portal_standard=True,
        tilt_offset=0.0,
        volume_offset={"x": 0.0, "y": 0.0, "z": 0.0},
        x_rotation_offset=0.0,
        per_section_alignment_parameters=params,
        volume_dimension=volume_dimension_a or {"x": 0.0, "y": 0.0, "z": 0.0},
    )
