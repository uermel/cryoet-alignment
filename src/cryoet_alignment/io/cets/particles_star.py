"""Particle STAR flavours ⇄ corner-anchored Å positions and particle→tomogram rotation matrices.

Three flavours, each pinned to the reader that consumes it (positions are returned in the **corner-anchored
Å frame** of the volume the star refers to; the CETS centring is applied by the caller through
``annotations.TomogramFrame``):

``warp`` — what ``WarpTools ts_export_particles --input_star`` reads (``ExportParticlesTiltseries.cs``):
    required ``rlnCoordinateX/Y/Z`` (:534-547) + ``rlnMicrographName`` else ``rlnTomoName`` (:212);
    ``px = coord − shift`` with ``shift = rlnOriginX/Y/Z`` (px, as is) or ``rlnOrigin*Angst / (rlnPixelSize |
    rlnImagePixelSize)`` per row (:585-668; the optics table is joined into the rows, :425-470);
    ``Å = px · coords_angpix`` (:308, ``--coords_angpix`` is mandatory: :122-124); Eulers pass through (:677-701).
``m`` — what ``MTools create_species --particles_relion`` reads (``CreateSpecies.cs``): the same columns, the
    coordinate pixel resolved as ``rlnDetectorPixelSize·1e4 / rlnMagnification`` → particles ``rlnImagePixelSize``
    → optics ``rlnImagePixelSize`` → ``--angpix_coords`` (:490-519), shifts ``rlnOrigin* · angpix_shifts`` (px) or
    ``rlnOrigin*Angst`` (:602-615, ``AngPixShifts = 1`` for ``*Angst``), series name normalised by basename (:483).
``relion5`` — RELION 5 tomo particles (``src/jaz/tomography/particle_set.cpp``): ``rlnCenteredCoordinateX/Y/ZAngst``
    (centred on ``rlnTomoSize/2`` tilt-series pixels, :589-604; ``tomogram_set.cpp:314``) or legacy ``rlnCoordinate*``
    in decentred tilt-series pixels (:605-616), ``rlnOrigin*Angst`` subtracted after rotation by the subtomogram
    matrix (:355-368, :381-395), ``M = A_subtomogram · A_particle`` (:399-425), pixel size from the optics group's
    ``rlnTomoTiltSeriesPixelSize`` (:655-660). The centred form needs the volume extent to reach the corner frame.

Orientation matrices are ``euler.zyz_to_matrix(rot, tilt, psi)`` (particle → tomogram) in every flavour.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import starfile

from cryoet_alignment.io.cets.euler import matrices_to_zyz, zyz_to_matrices

__all__ = [
    "FLAVOURS",
    "ParticleTable",
    "read_particle_star",
    "write_particle_star",
    "series_stem",
    "detect_flavour",
    "PRESERVED_COLUMNS",
]

FLAVOURS = ("warp", "m", "relion5")

#: Columns other than geometry that are carried into the CETS companion and written back (Warp passes any
#: unknown column through unchanged, ExportParticlesTiltseries.cs:185-210).
PRESERVED_COLUMNS = (
    "rlnRandomSubset",
    "rlnClassNumber",
    "rlnGroupNumber",
    "rlnTomoParticleId",
    "rlnTomoParticleName",
    "rlnOpticsGroup",
    "rlnAutopickFigureOfMerit",
    "rlnMaxValueProbDistribution",
    "rlnLogLikeliContribution",
)

_GEOMETRY_COLUMNS = {
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnCenteredCoordinateXAngst",
    "rlnCenteredCoordinateYAngst",
    "rlnCenteredCoordinateZAngst",
    "rlnOriginX",
    "rlnOriginY",
    "rlnOriginZ",
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnOriginZAngst",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnTomoSubtomogramRot",
    "rlnTomoSubtomogramTilt",
    "rlnTomoSubtomogramPsi",
    "rlnMicrographName",
    "rlnTomoName",
    "rlnPixelSize",
    "rlnImagePixelSize",
    "rlnDetectorPixelSize",
    "rlnMagnification",
    "rlnOpticsGroupName",
}

_SERIES_SUFFIXES = (".tomostar", ".mrc", ".mrcs", ".st", ".xml")


def series_stem(name: str) -> str:
    """``../tomostar/TS_01.tomostar`` → ``TS_01`` (basename, known suffix stripped) — ``create_species`` matches
    by basename (CreateSpecies.cs:483), ``ts_export_particles`` by the exact ``<stem>.tomostar`` (Movie.cs:47)."""
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    low = base.lower()
    for suf in _SERIES_SUFFIXES:
        if low.endswith(suf):
            return base[: -len(suf)]
    return base


@dataclass
class ParticleTable:
    flavour: str
    path: Optional[Path]
    series: List[str]  # normalised stem per row
    series_raw: List[str]  # the name column as written
    positions_corner_a: np.ndarray  # (N,3) corner-anchored Å
    matrices: Optional[np.ndarray]  # (N,3,3) particle→tomogram, None when the star has no Euler columns
    coords_angpix: Optional[float]
    coords_angpix_source: str
    angpix_shifts: Optional[float] = None
    tilt_series_pixel_a: Optional[float] = None  # relion5
    centred_input: bool = False  # relion5: rlnCenteredCoordinate*Angst read (positions relative to extent/2)
    extra: Dict[str, list] = field(default_factory=dict)  # preserved columns
    dropped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.positions_corner_a))

    def rows_for(self, stem: str) -> np.ndarray:
        return np.array([i for i, s in enumerate(self.series) if s == stem], dtype=int)


# --------------------------------------------------------------------------- reading


def _load_blocks(path: Union[str, Path]) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, str]:
    blocks = starfile.read(str(path), always_dict=True)
    optics = None
    particles = None
    pname = ""
    for name, block in blocks.items():
        if not isinstance(block, pd.DataFrame):
            continue
        cols = set(block.columns)
        if name == "optics" and "rlnOpticsGroup" in cols:
            optics = block
        elif cols & {"rlnCoordinateX", "rlnCenteredCoordinateXAngst"}:
            particles, pname = block, name
    if particles is None:
        raise ValueError(f"{path}: no block with rlnCoordinateX / rlnCenteredCoordinateXAngst")
    return optics, particles.reset_index(drop=True), pname


def _per_row_from_optics(particles: pd.DataFrame, optics: Optional[pd.DataFrame], column: str) -> Optional[np.ndarray]:
    """A column per particle row: from the particles table, else joined from the optics table by rlnOpticsGroup."""
    if column in particles.columns:
        return particles[column].to_numpy(dtype=np.float64)
    if optics is not None and column in optics.columns:
        if "rlnOpticsGroup" in particles.columns and "rlnOpticsGroup" in optics.columns:
            lut = {int(g): float(v) for g, v in zip(optics["rlnOpticsGroup"].tolist(), optics[column].tolist())}
            groups = particles["rlnOpticsGroup"].astype(int).to_numpy()
            missing = sorted(set(groups) - set(lut))
            if missing:
                raise ValueError(f"optics group(s) {missing} referenced by particles are not in the optics table")
            return np.array([lut[g] for g in groups], dtype=np.float64)
        return np.full(len(particles), float(optics[column].iloc[0]), dtype=np.float64)
    return None


def detect_flavour(particles: pd.DataFrame, optics: Optional[pd.DataFrame]) -> str:
    cols = set(particles.columns)
    if {"rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"} <= cols:
        return "relion5"
    if "rlnTomoName" in cols and "rlnMicrographName" not in cols:
        has_ts_pix = "rlnTomoTiltSeriesPixelSize" in cols or (
            optics is not None and "rlnTomoTiltSeriesPixelSize" in optics.columns
        )
        if has_ts_pix:
            return "relion5"
    if "rlnMicrographName" in cols or "rlnTomoName" in cols:
        return "warp"
    raise ValueError(
        "cannot detect the star flavour: no rlnMicrographName / rlnTomoName / rlnCenteredCoordinate* columns",
    )


def _name_column(particles: pd.DataFrame, flavour: str) -> str:
    if flavour == "relion5":
        for c in ("rlnTomoName", "rlnMicrographName"):
            if c in particles.columns:
                return c
    else:  # Warp prefers rlnMicrographName (ExportParticlesTiltseries.cs:212), M likewise (CreateSpecies.cs:470-478)
        for c in ("rlnMicrographName", "rlnTomoName"):
            if c in particles.columns:
                return c
    raise ValueError("no rlnMicrographName / rlnTomoName column: rows cannot be assigned to a tilt series")


def _eulers(particles: pd.DataFrame) -> Optional[np.ndarray]:
    cols = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    if not all(c in particles.columns for c in cols):
        return None
    return zyz_to_matrices(particles[list(cols)].to_numpy(dtype=np.float64))


def _preserved(particles: pd.DataFrame) -> Tuple[Dict[str, list], List[str]]:
    extra: Dict[str, list] = {}
    dropped: List[str] = []
    for c in particles.columns:
        if c in _GEOMETRY_COLUMNS:
            continue
        if c in PRESERVED_COLUMNS:
            extra[c] = particles[c].tolist()
        else:
            dropped.append(c)
    return extra, dropped


def read_particle_star(
    path: Union[str, Path],
    flavour: str = "auto",
    *,
    coords_angpix: Optional[float] = None,
    angpix_shifts: Optional[float] = None,
) -> ParticleTable:
    """Read a particle star in one of the pinned flavours into corner-anchored Å positions and matrices.

    ``coords_angpix`` / ``angpix_shifts`` override what the star declares (the CLI's explicit values). For
    ``relion5`` centred coordinates the positions are relative to ``extent/2`` (``centred_input`` is set) and the
    caller adds the extent of the alignment's box to reach the corner frame.
    """
    path = Path(path)
    optics, particles, _ = _load_blocks(path)
    if flavour == "auto":
        flavour = detect_flavour(particles, optics)
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown star flavour {flavour!r} (one of {FLAVOURS})")
    name_col = _name_column(particles, flavour)
    raw = [str(v) for v in particles[name_col].tolist()]
    stems = [series_stem(v) for v in raw]
    n = len(particles)
    notes: List[str] = []
    matrices = _eulers(particles)

    if flavour == "relion5":
        ts_pix = _per_row_from_optics(particles, optics, "rlnTomoTiltSeriesPixelSize")
        if coords_angpix is not None:
            ts_pix = np.full(n, float(coords_angpix))
            src = "explicit"
        elif ts_pix is None:
            raise ValueError(
                f"{path.name}: rlnTomoTiltSeriesPixelSize missing from the optics table; pass the tilt-series pixel size",
            )
        else:
            src = "rlnTomoTiltSeriesPixelSize"
        a_sub = None
        if all(
            c in particles.columns for c in ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi")
        ):
            a_sub = zyz_to_matrices(
                particles[["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"]].to_numpy(
                    dtype=np.float64,
                ),
            )
            if matrices is not None:
                matrices = np.einsum("nij,njk->nik", a_sub, matrices)  # A_subtomogram · A_particle
        origin = np.zeros((n, 3))
        if all(c in particles.columns for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")):
            origin = particles[["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]].to_numpy(dtype=np.float64)
            if a_sub is not None:
                origin = np.einsum("nij,nj->ni", a_sub, origin)
        centred = all(
            c in particles.columns
            for c in ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst")
        )
        if centred:
            pos = particles[
                ["rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"]
            ].to_numpy(
                dtype=np.float64,
            )
        else:
            pos = (
                particles[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=np.float64)
                * ts_pix[:, None]
            )
        pos = pos - origin
        extra, dropped = _preserved(particles)
        return ParticleTable(
            flavour="relion5",
            path=path,
            series=stems,
            series_raw=raw,
            positions_corner_a=pos,
            matrices=matrices,
            coords_angpix=float(ts_pix[0]),
            coords_angpix_source=src,
            tilt_series_pixel_a=float(ts_pix[0]),
            centred_input=centred,
            extra=extra,
            dropped=dropped,
            notes=notes,
        )

    # warp / m: the same columns; the pixel-size chain differs only in what is accepted without a flag
    coords = particles[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=np.float64)
    if coords_angpix is not None:
        a = float(coords_angpix)
        src = "explicit"
    else:
        det = _per_row_from_optics(particles, optics, "rlnDetectorPixelSize")
        mag = _per_row_from_optics(particles, optics, "rlnMagnification")
        ips = _per_row_from_optics(particles, optics, "rlnImagePixelSize")
        if det is not None and mag is not None:
            vals = det * 1e4 / mag
            src = "rlnDetectorPixelSize*1e4/rlnMagnification"
        elif ips is not None:
            vals = ips
            src = "rlnImagePixelSize"
        else:
            raise ValueError(
                f"{path.name}: no coordinate pixel size in the star (rlnImagePixelSize / rlnDetectorPixelSize); "
                "pass --coords-angpix",
            )
        if np.ptp(vals) > 1e-6 * abs(vals[0]):
            raise ValueError(
                f"{path.name}: {src} varies between rows ({vals.min()} .. {vals.max()}); pass --coords-angpix",
            )
        a = float(vals[0])
    # shifts: px columns as is (Warp) / × angpix_shifts (M, default = coords pixel); *Angst / per-row pixel size
    shift_px = np.zeros((n, 3))
    if all(c in particles.columns for c in ("rlnOriginX", "rlnOriginY", "rlnOriginZ")):
        s = particles[["rlnOriginX", "rlnOriginY", "rlnOriginZ"]].to_numpy(dtype=np.float64)
        # M: rlnOrigin* × AngPixShifts (CreateSpecies.cs:602-609); Warp: pixels as they are (:630-634)
        shift_px = s * float(angpix_shifts) / a if flavour == "m" and angpix_shifts is not None else s
        angpix_shifts_used = float(angpix_shifts) if angpix_shifts is not None else a
    elif all(c in particles.columns for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")):
        s = particles[["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]].to_numpy(dtype=np.float64)
        px = _per_row_from_optics(particles, optics, "rlnPixelSize")
        if px is None:
            px = _per_row_from_optics(particles, optics, "rlnImagePixelSize")
        if px is None:
            raise ValueError(
                f"{path.name}: rlnOrigin*Angst without rlnPixelSize / rlnImagePixelSize (Warp refuses this too)",
            )
        shift_px = s / px[:, None]
        angpix_shifts_used = 1.0
    else:
        angpix_shifts_used = None
    pos = (coords - shift_px) * a
    extra, dropped = _preserved(particles)
    return ParticleTable(
        flavour=flavour,
        path=path,
        series=stems,
        series_raw=raw,
        positions_corner_a=pos,
        matrices=matrices,
        coords_angpix=a,
        coords_angpix_source=src,
        angpix_shifts=angpix_shifts_used,
        extra=extra,
        dropped=dropped,
        notes=notes,
    )


# --------------------------------------------------------------------------- writing


@dataclass
class StarRows:
    """What a writer needs per particle: series stem, corner-anchored Å position, optional matrix, extras."""

    series: List[str]
    positions_corner_a: np.ndarray  # (N,3)
    matrices: Optional[np.ndarray]  # (N,3,3) or None
    extra: Dict[str, list] = field(default_factory=dict)


def _series_name(stem: str, style: str) -> str:
    if style == "tomostar":
        return f"{stem}.tomostar"
    if style == "stem":
        return stem
    raise ValueError(f"unknown series name style {style!r} (tomostar|stem)")


def write_particle_star(
    path: Union[str, Path],
    flavour: str,
    rows: StarRows,
    *,
    coords_angpix: float,
    series_name_style: str = "tomostar",
    extent_a: Optional[Sequence[float]] = None,
    voltage_kv: Optional[float] = None,
    cs_mm: Optional[float] = None,
    amplitude_contrast: Optional[float] = None,
) -> Path:
    """Write the column set the target reader needs (see module docstring).

    ``warp``: ``rlnMicrographName``, ``rlnCoordinateX/Y/Z`` (px of ``coords_angpix``), ``rlnOriginX/Y/Z = 0``,
    Eulers when matrices are given. ``m``: the same plus ``rlnOpticsGroup`` and a ``data_optics`` block with
    ``rlnImagePixelSize = coords_angpix`` (the M pixel-size chain) and ``rlnOrigin*Angst = 0`` (so ``AngPixShifts = 1``).
    ``relion5``: ``rlnTomoName``, ``rlnCenteredCoordinateX/Y/ZAngst`` (= corner Å − extent/2), ``rlnOrigin*Angst = 0``,
    ``rlnTomoSubtomogram* = 0`` when Eulers are written, optics ``rlnTomoTiltSeriesPixelSize = coords_angpix``.
    """
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown star flavour {flavour!r} (one of {FLAVOURS})")
    path = Path(path)
    pos = np.asarray(rows.positions_corner_a, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    if len(rows.series) != n:
        raise ValueError(f"{len(rows.series)} series names for {n} positions")
    names = [_series_name(s, series_name_style) for s in rows.series]
    eulers = matrices_to_zyz(rows.matrices) if rows.matrices is not None else None
    data: Dict[str, list] = {}
    if flavour == "relion5":
        if extent_a is None:
            raise ValueError("relion5 flavour needs the volume extent (Å) to centre the coordinates")
        ext = np.asarray(extent_a, dtype=np.float64).reshape(3)
        centred = pos - ext / 2.0
        data["rlnTomoName"] = names
        data["rlnCenteredCoordinateXAngst"] = centred[:, 0].tolist()
        data["rlnCenteredCoordinateYAngst"] = centred[:, 1].tolist()
        data["rlnCenteredCoordinateZAngst"] = centred[:, 2].tolist()
        for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"):
            data[c] = [0.0] * n
        if eulers is not None:
            data["rlnAngleRot"], data["rlnAngleTilt"], data["rlnAnglePsi"] = (eulers[:, i].tolist() for i in range(3))
            for c in ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"):
                data[c] = [0.0] * n
        data["rlnOpticsGroup"] = [1] * n
    else:
        px = pos / float(coords_angpix)
        data["rlnMicrographName"] = names
        data["rlnCoordinateX"] = px[:, 0].tolist()
        data["rlnCoordinateY"] = px[:, 1].tolist()
        data["rlnCoordinateZ"] = px[:, 2].tolist()
        if eulers is not None:
            data["rlnAngleRot"], data["rlnAngleTilt"], data["rlnAnglePsi"] = (eulers[:, i].tolist() for i in range(3))
        if flavour == "warp":
            for c in ("rlnOriginX", "rlnOriginY", "rlnOriginZ"):
                data[c] = [0.0] * n
        else:  # m: *Angst origins so AngPixShifts = 1 (CreateSpecies.cs:508-509)
            for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"):
                data[c] = [0.0] * n
            data["rlnOpticsGroup"] = [1] * n
    for col, values in rows.extra.items():
        if col in data or col not in PRESERVED_COLUMNS:
            continue
        if len(values) != n:
            raise ValueError(f"preserved column {col!r} has {len(values)} values for {n} rows")
        data[col] = list(values)
    particles = pd.DataFrame(data)

    blocks: Dict[str, pd.DataFrame] = {}
    if flavour in ("m", "relion5"):
        optics: Dict[str, list] = {"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["opticsGroup1"]}
        if flavour == "m":
            optics["rlnImagePixelSize"] = [float(coords_angpix)]
        else:
            optics["rlnTomoTiltSeriesPixelSize"] = [float(coords_angpix)]
            optics["rlnImagePixelSize"] = [float(coords_angpix)]
        if voltage_kv is not None:
            optics["rlnVoltage"] = [float(voltage_kv)]
        if cs_mm is not None:
            optics["rlnSphericalAberration"] = [float(cs_mm)]
        if amplitude_contrast is not None:
            optics["rlnAmplitudeContrast"] = [float(amplitude_contrast)]
        blocks["optics"] = pd.DataFrame(optics)
        blocks["particles"] = particles
    else:
        blocks["particles"] = particles
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    starfile.write(blocks, str(path), float_format="%.6f")
    return path


def validate_series_names(table: ParticleTable, known_stems: Sequence[str]) -> List[str]:
    """Raw name-column values whose stem is not in ``known_stems`` (empty when every row binds)."""
    known = set(known_stems)
    bad = sorted({r for r, s in zip(table.series_raw, table.series) if s not in known})
    return bad


_STAR_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_stem(name: str) -> str:
    return _STAR_NAME_RE.sub("_", Path(name).stem)
