# CETS interoperability profile `cets-rigid/0.2`

`cryoet_alignment.io.cets` reads and writes exactly the subset of CETS (TomoBabel cryo-ET standard) documents
described here. Anything else is rejected with the first mismatching element named, or handled by a named
adapter (`io/cets/adapters/`). The schema is pinned to cets-data-models commit
`b415e952d309ac1ec0dee01a3a5db639cb08bba5` (PR #34 head, contains `main`; projection sequences of up to three
entries):

    pip install 'cets_data_model @ git+https://github.com/TomoBabel/cets-data-models.git@b415e952d309ac1ec0dee01a3a5db639cb08bba5'

## Entities

- `TiltSeries{id, path, images[TiltImage], movie_stack_series_id, ctf_corrected}` with one `TiltImage` per
  **raw section**: `id = <ts>_<section>`, `section` = raw z index, `nominal_tilt_angle` = the stage angle as
  acquired (never a refined value), `accumulated_dose` = exclusive pre-exposure, `ctf_metadata` when known,
  `width`/`height`, frames (below). A section without a projection alignment is dark / excluded.
- `Tomogram{id, path, tilt_series_id, width, height, depth, ctf_corrected}` with frames (voxel size).
- `Alignment{tilt_series_id, projection_alignments[]}` under `Region.alignments`; `Alignment` has no id, so the
  projection-alignment ids carry an **alignment-instance name**: `<ts>_<name>_align_<section>`
  (`aretomo3`, `warp`, `portal18924`, …). Several alignments of one tilt series stay distinct.
- Companion manifest `<doc>.cets-companion.json` (schema `cets-rigid-companion/0.2`, `io/cets/companion.py`; 0.1 companions still load):
  explicitly outside CETS — acquisition order, per-image exposure, nominal-vs-refined provenance, kV / Cs /
  amplitude contrast, defocus hand, AlphaOffset / BetaOffset, `AreAnglesInverted`, pixel sizes, FlipVol,
  alignment ↔ tomogram bindings, header vs implied voxel sizes, dropped items.

## Frames

Every image entity carries the coordinate systems `array` (axis_type `array`, unit `pixel`/`voxel`, corner
origin, integer indices) and `physical` (axis_type `space`, unit `angstrom`) and **exactly one** transform named
`array_to_physical`:

    Sequence[Translation("center_array_origin", -floor(N/2) per axis), Scale("scale_to_physical", s per axis)]

so the physical origin sits at array index `floor(N/2)` (CETS issue #1, PR #34). Readers validate the chain
(`frames.image_frame`): a bare `Scale` is accepted and means a **corner-origin** physical frame — never an
implied centering; anisotropic spacing, extra transforms of that name, or other chain shapes are rejected.

## Projection alignment

    ProjectionAlignment(id=<ts>_<name>_align_<z>, tilt_image_id=<ts>_<z>, name="tomogram_to_projection",
                        input="physical", output="physical",
                        sequence=[Affine("tilt",              Ry(tilt) · Rx(xrot)),   # 3x3
                                  Affine("in_plane_rotation", Rz(rot)),               # 3x3
                                  Translation("shift",        [tx, ty])])             # Å

    q_img = drop_z( Rz(rot) · Ry(tilt) · Rx(xrot) · p_tomo ) + shift

`p_tomo` is a point in the **reference tomogram's** centred physical frame (origin at array index `floor(N/2)`
of that tomogram; z axis = Warp/RELION z), `q_img` the point in the tilt image's centred physical frame. The
matrices are the standard right-handed active rotations of RELION's `t3Matrix::rotation` port
(`io.relion.alignment.rot_x/rot_y/rot_z`, `Ry(θ): x' = x cos θ + z sin θ`); `rot` = tilt-axis rotation
(AreTomo3 `ROT`, RELION `rlnTomoZRot`, Warp `AxisAngle`), `tilt` = refined tilt (`TILT`, `rlnTomoYTilt`,
`-(Angle + LevelAngleY)`), `xrot` = `rlnTomoXTilt` / `LevelAngleX`. Readers fold the sequence generically
(affines multiply, translations add, in list order), so the two-entry `[Affine R, Translation]` form and
reordered names decode to the same operator; affines must be proper rotations (2D-homogeneous affines with a
translation column are rejected).

## Frame contract (native ↔ CETS)

The hub model (`io.cryoet_data_portal.Alignment`, AreTomo3-convention parameters, offsets in tilt-image
pixels) refers to the native tool's centred frames. Their origins differ from CETS for odd sizes:

| source (`Alignment.format`) | image centre | volume centre |
|---|---|---|
| `ARETOMO3` | `N/2` (float) | `V/2` (float) |
| `WARP` | `N/2` | `V/2` |
| `RELION` | `floor(N/2)` | `floor(D/2)` (matrix, `tomogram.cpp:44,53`) |
| `IMOD` | not pinned — even sizes only through the adapter | |
| CETS `physical` | `floor(N/2)` | `floor(N/2)` of the reference tomogram |

With `delta = native_centre − cets_centre` (Å, per axis) the codec applies, on BOTH directions,

    shift_cets = shift_native + delta_image − (R · delta_volume)[:2]

(`alignment_to_cets` / `alignment_from_cets` take the `FrameConvention`, the tilt-image frame and the
reference tomogram explicitly; nothing is left to callers). Gates: `tests/test_cets_goldens.py` (odd sizes vs
arewarpion's `AretomoTsModel`, ≤ 1e-9 px; the naive recipe is off by ~1.6 px), `tests/test_cets_warp_goldens.py`
(vs `WarpTiltSeriesModel` with constant grids, ≤ 1e-2 Å), `tests/test_cets_relion_golden.py` (exact).

### Reference tomogram

Each encoded alignment is bound to one reference tomogram; the hub's native volume box (`volume_dimension`,
Å) must agree with `N_tomo · s_tomo` within one voxel per axis or the association is refused. Portal
tomograms must use the **implied** voxel `s_ts · N_ts / N_tomo` (the header voxel is rounded); the companion
records both. Selection is explicit whenever a region has several alignments or tomograms.

## Warp specifics

- Constant deformation grids are rigid contributions and are folded: `GridMovement` (subtracted after
  projection) into the effective shift, a constant `GridVolumeWarp` (added before rotation) into `R·w`.
  Spatially varying grids are refused unless dropped explicitly; non-zero `GridAngleX/Y/Z` are reported
  (they affect particle orientations).
- `LevelAngleX` is the common `xrot`; `LevelAngleY` is a gauge (written 0; recorded as `tilt_offset = -LevelAngleY`).
  Per-section `xrot` values must agree within `1e-3°` to be written to Warp; AreTomo3 cannot represent `xrot`
  at all (`to_aretomo` refuses).
- `AreAnglesInverted` changes only Warp's defocus/depth channel, never XY; it is provenance (companion).
- One XML row per tilt series row: rows without an alignment are written with `UseTilt=False`.
- `CTF.PixelSize` is the sampling Warp used for CTF fitting (`BinnedPixelSizeMean`), not necessarily the
  stored tilt-image pixel; the tilt-image frame comes from the image header.

## Units and nulls

Angles in degrees; defocus in Å underfocus-positive with `defocus_u ≥ defocus_v`, `defocus_angle` in
`[0, 180)` from +x toward +y, `phase_shift` in degrees (AreTomo3/portal radians and Warp π-units are converted at
the boundary). Unknown `defocus_handedness` is written as an explicit `null` — the model would otherwise
materialise `-1` — and documents are dumped without `exclude_none`. `_CTF.txt` score / resolution are not
preserved (placeholders `0.0` / `999.99` on write).

## What round-trips through CETS alone, and what needs the companion

| | CETS alone | with companion |
|---|---|---|
| rotation operators, shifts, CTF values, dark/excluded sections | yes | |
| acquisition order, per-image exposure (`[2,3,100]` → pre-exposures `[0,2,5]` lose the last), nominal vs refined angles | no | yes |
| kV, Cs, amplitude contrast, defocus hand, AlphaOffset / BetaOffset, `AreAnglesInverted`, FlipVol | no | yes |
| Warp / AreTomo3 locals, `GridAngle*`, `Thickness`, `MagnificationCorrection`, portal registration matrices | no (reported as dropped / refused) | recorded |
| point / oriented-point annotations (positions, particle → tomogram matrices), mask geometry | yes | |
| annotation provenance (portal metadata, object, method), star pixel sizes, preserved star columns, mask labels | no | yes |

## Annotations (profile 0.2)

An annotation binds to exactly one tomogram entity through `source_tomogram_id` (required by the profile;
the schema leaves it optional). Its coordinates are Å in **that tomogram's centred physical frame**: origin at
array index `floor(N/2)` of the tomogram, corner-anchored voxel grid (voxel `k` at `k·s`), z = tomogram z.
The entity declares `coordinate_systems = [physical]` and one transform named `annotation_to_tomogram`
(`physical` → `physical`), the identity when written by this codec. Readers fold any such chain of
`Identity` / `Translation` / `Scale` / `Affine` / `Sequence` into `(A, t)` and apply `p_tomo = A·p + t`
(orientations are rotated by the rotational part, uniform scale allowed); other names, systems or steps are
refused naming the element. Grid offsets between reconstruction engines (e.g. AreTomo3's kernel conventions
against Warp's/RELION's corner-anchored grid) are **not modelled**.

| entity | content |
|---|---|
| `PointSet3D` | `origin3D[i] = [x, y, z]` Å |
| `PointMatrixSet3D` | `origin3D` + `matrix3D[i]` = the active rotation mapping particle (reference map) coordinates to tomogram coordinates, `p_tomo = M·p_map + origin`; = RELION's `A(rot, tilt, psi)` (`Euler::anglesToMatrix3`, used un-transposed by `ParticleSet::getMatrix4x4`) = Warp's `Matrix3.Euler` (`ParticleMatrix`) = the cryoET Data Portal's `xyz_rotation_matrix` |
| `SegmentationMask3D` | `path`, `width/height/depth`, its own `array`/`physical` frame (`array_to_physical` for the mask grid) plus the identity `annotation_to_tomogram`; label semantics in the companion |

Ids: `<tomogram_id>_ann_<key>` (portal: `<annotation id>_<shape>`; stars: the star stem), unique per region.

**Particle stars** (`io/cets/particles_star.py`). Positions are exchanged through the corner-anchored Å frame of
the bound tomogram (`c = floor(N/2)·s`, `E = N·s`, `h = E/2 − c`):

| flavour | consumer | read | write |
|---|---|---|---|
| `warp` | `WarpTools ts_export_particles --input_star` | `px = rlnCoordinate* − rlnOrigin*` (px) or `− rlnOrigin*Angst / (rlnPixelSize\|rlnImagePixelSize)` per row; `Å = px · coords_angpix` (mandatory, as for Warp); series = `rlnMicrographName` else `rlnTomoName`; Eulers pass through | `rlnMicrographName = <stem>.tomostar`, `rlnCoordinate* = (p + c)/a`, `rlnOrigin* = 0`, `rlnAngleRot/Tilt/Psi` from `M` |
| `m` | `MTools create_species --particles_relion` | same columns; coordinate pixel `rlnDetectorPixelSize·1e4/rlnMagnification` → `rlnImagePixelSize` (particles, then optics) → `--angpix_coords`; `rlnOrigin* · angpix_shifts` or `*Angst` | as `warp` + `data_optics{rlnImagePixelSize = a}` and `rlnOrigin*Angst = 0` |
| `relion5` | RELION 5 tomo | `rlnCenteredCoordinate*Angst` (from the float centre `E/2`) or legacy `rlnCoordinate* · rlnTomoTiltSeriesPixelSize`; `− A_subtomogram · rlnOrigin*Angst`; `M = A_subtomogram · A_particle` | `rlnTomoName`, `rlnCenteredCoordinate*Angst = p_corner − E/2`, `rlnOrigin*Angst = 0`, `rlnTomoSubtomogram* = 0`, optics `rlnTomoTiltSeriesPixelSize = a` |

`warp` and `m` share their columns, so auto-detection reports `warp` for both; the same file can legitimately
mean different positions to M and to RELION 5 when `rlnImagePixelSize ≠ rlnTomoTiltSeriesPixelSize` (verified
against an M species table: M used `rlnImagePixelSize`). Preserved columns (`rlnRandomSubset`, `rlnClassNumber`,
`rlnGroupNumber`, `rlnTomoParticleId/Name`, `rlnOpticsGroup`, figure-of-merit columns) travel in the companion and
are written back; other columns are listed as dropped.

**Portal ndjson**: `location` is a voxel index at the annotation file's voxel spacing, corner origin, no
half-voxel term; `p = (loc − floor(N/2))·s` with the grid of the portal tomogram the file is attached to;
`xyz_rotation_matrix` is `M` itself (the backend stores `Rotation.from_euler("ZYZ", …).inv()`, which equals
`anglesToMatrix3`).
