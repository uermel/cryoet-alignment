"""Builders for the CETS entities the profile exchanges, and JSON I/O.

``TiltSeries.images`` lists EVERY raw section (``section = z_index``); ``nominal_tilt_angle`` is the stage
angle as acquired (never a refined value); ``accumulated_dose`` the exclusive pre-exposure; images without a
``ProjectionAlignment`` are dark/excluded. Documents are dumped with ``model_dump(mode="json")`` and
WITHOUT ``exclude_none`` so explicit nulls (``defocus_handedness``) survive.
"""

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

from cryoet_alignment.io.cets.frames import attach_frames
from cryoet_alignment.io.cets.profile import require_cets, tilt_image_id

PATH = Union[str, Path]


def tilt_series_entity(
    *,
    tilt_series_id: str,
    path: Optional[str],
    width: int,
    height: int,
    pixel_size_a: float,
    nominal_angles: Sequence[float],
    doses: Optional[Sequence[Optional[float]]] = None,
    ctfs: Optional[Sequence[Optional[object]]] = None,
    movie_stack_ids: Optional[Sequence[Optional[str]]] = None,
    image_paths: Optional[Sequence[Optional[str]]] = None,
    movie_stack_series_id: Optional[str] = None,
    ctf_corrected: bool = False,
    even_path: Optional[str] = None,
    odd_path: Optional[str] = None,
):
    """A ``TiltSeries`` with one ``TiltImage`` per raw section, frames attached."""
    m = require_cets()
    n = len(nominal_angles)

    def _at(seq, i):
        return None if seq is None else seq[i]

    images = []
    for i in range(n):
        im = m.TiltImage(
            id=tilt_image_id(tilt_series_id, i),
            movie_stack_id=_at(movie_stack_ids, i),
            path=_at(image_paths, i) if image_paths is not None else path,
            section=i,
            nominal_tilt_angle=float(nominal_angles[i]),
            accumulated_dose=None if _at(doses, i) is None else float(_at(doses, i)),
            ctf_metadata=_at(ctfs, i),
            width=int(width),
            height=int(height),
        )
        attach_frames(im, (int(width), int(height)), (pixel_size_a, pixel_size_a))
        images.append(im)
    return m.TiltSeries(
        id=tilt_series_id,
        path=path,
        ctf_corrected=ctf_corrected,
        even_path=even_path,
        odd_path=odd_path,
        movie_stack_series_id=movie_stack_series_id,
        images=images,
    )


def tomogram_entity(
    *,
    tomogram_id: str,
    path: Optional[str],
    size_px: Sequence[int],
    voxel_size_a: Union[float, Sequence[float]],
    tilt_series_id: str,
    ctf_corrected: bool = False,
    even_path: Optional[str] = None,
    odd_path: Optional[str] = None,
):
    m = require_cets()
    if isinstance(voxel_size_a, (int, float)):
        voxel = (float(voxel_size_a),) * 3
    else:
        voxel = tuple(float(v) for v in voxel_size_a)
    tomo = m.Tomogram(
        id=tomogram_id,
        path=path,
        tilt_series_id=tilt_series_id,
        ctf_corrected=ctf_corrected,
        even_path=even_path,
        odd_path=odd_path,
        width=int(size_px[0]),
        height=int(size_px[1]),
        depth=int(size_px[2]),
    )
    attach_frames(tomo, tuple(int(v) for v in size_px), voxel)
    return tomo


def movie_stack_series_entity(*, series_id: str, stacks: Sequence[dict]):
    """``MovieStackSeries`` from ``[{"id", "path", "n_frames"?, "width"?, "height"?, "pixel_size_a"?}]``."""
    m = require_cets()
    out = []
    for st in stacks:
        frames = []
        for f in range(int(st.get("n_frames", 0) or 0)):
            fr = m.MovieFrame(path=st.get("path"), section=f)
            if st.get("width") and st.get("height") and st.get("pixel_size_a"):
                fr.width, fr.height = int(st["width"]), int(st["height"])
                attach_frames(fr, (fr.width, fr.height), (st["pixel_size_a"], st["pixel_size_a"]))
            frames.append(fr)
        out.append(m.MovieStack(id=st["id"], path=st.get("path"), images=frames))
    return m.MovieStackSeries(id=series_id, stacks=out)


def region_entity(
    *,
    region_id: str,
    tilt_series: Sequence[object] = (),
    alignments: Sequence[object] = (),
    tomograms: Sequence[object] = (),
    movie_stack_series: Sequence[object] = (),
    annotations: Sequence[object] = (),
    gain_file=None,
    defect_file=None,
):
    m = require_cets()
    collection = None
    if movie_stack_series or gain_file or defect_file:
        collection = m.MovieStackCollection(
            movie_stacks=list(movie_stack_series), gain_file=gain_file, defect_file=defect_file,
        )
    return m.Region(
        id=region_id,
        movie_stack_collection=collection,
        tilt_series=list(tilt_series),
        alignments=list(alignments),
        tomograms=list(tomograms),
        annotations=list(annotations),
    )


def dataset_entity(name: str, regions: Sequence[object]):
    m = require_cets()
    return m.Dataset(name=name, regions=list(regions))


def to_json_dict(model) -> dict:
    return model.model_dump(mode="json")


def dump_json(model, path: PATH, indent: int = 2) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_json_dict(model), indent=indent) + "\n")
    return path


def load_dataset(path: PATH):
    m = require_cets()
    return m.Dataset.model_validate(json.loads(Path(path).read_text()))


def validate_document(model) -> None:
    """Round-trip through the pinned schema (raises ``pydantic.ValidationError`` on any violation)."""
    type(model).model_validate(to_json_dict(model))


__all__: List[str] = [
    "dataset_entity",
    "dump_json",
    "load_dataset",
    "movie_stack_series_entity",
    "region_entity",
    "tilt_series_entity",
    "to_json_dict",
    "tomogram_entity",
    "validate_document",
]
