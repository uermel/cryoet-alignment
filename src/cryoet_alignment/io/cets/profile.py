"""Names and versions that define the ``cets-rigid/0.1`` interoperability profile.

The profile is the exact subset of CETS documents this codec reads and writes; anything else is rejected
with the first mismatching element named (or routed through an explicit adapter). See ``docs/cets.md``.
"""

PROFILE_VERSION = "cets-rigid/0.1"
COMPANION_VERSION = "cets-rigid-companion/0.1"

#: Pinned cets-data-models commit (PR #34 head; contains main, sequence cardinality 3).
CETS_DATA_MODEL_COMMIT = "b415e952d309ac1ec0dee01a3a5db639cb08bba5"
CETS_INSTALL_HINT = (
    f"pip install 'cets_data_model @ git+https://github.com/TomoBabel/cets-data-models.git@{CETS_DATA_MODEL_COMMIT}'"
)

# Coordinate system names (PR #34 CoordinateSpaceName)
ARRAY_CS = "array"
PHYSICAL_CS = "physical"

# Transformation names (PR #34 TransformationName + the two sub-names of its example)
ARRAY_TO_PHYSICAL = "array_to_physical"
CENTER_ARRAY_ORIGIN = "center_array_origin"
SCALE_TO_PHYSICAL = "scale_to_physical"
TOMOGRAM_TO_PROJECTION = "tomogram_to_projection"
ANNOTATION_TO_TOMOGRAM = "annotation_to_tomogram"
TILT = "tilt"
IN_PLANE_ROTATION = "in_plane_rotation"
SHIFT = "shift"

# Axis units (strings valid on cets-data-models main and as the PR #34 AxisUnit enum values)
UNIT_PIXEL = "pixel"
UNIT_VOXEL = "voxel"
UNIT_ANGSTROM = "angstrom"


def require_cets():
    """Import and return ``cets_data_model.models.models`` or raise an actionable ImportError."""
    try:
        from cets_data_model.models import models
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise ImportError(
            "cryoet_alignment.io.cets needs the CETS data model package; install it with\n  " + CETS_INSTALL_HINT,
        ) from exc
    return models


def tilt_image_id(tilt_series_id: str, section: int) -> str:
    """``<ts>_<section>`` (cets-imod / PR #34 convention)."""
    return f"{tilt_series_id}_{section}"


def projection_alignment_id(tilt_series_id: str, alignment_name: str, section: int) -> str:
    """``<ts>_<alignment>_align_<section>`` — the alignment-instance name keeps several alignments of one
    tilt series distinct (``Alignment`` itself has no id)."""
    return f"{tilt_series_id}_{alignment_name}_align_{section}"


def section_from_tilt_image_id(tilt_image_id_: str, tilt_series_id: str) -> int:
    prefix = f"{tilt_series_id}_"
    if not tilt_image_id_.startswith(prefix):
        raise ValueError(f"tilt image id {tilt_image_id_!r} does not belong to tilt series {tilt_series_id!r}")
    return int(tilt_image_id_[len(prefix) :])
