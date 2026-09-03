# cryoet-alignment

Convert between different alignment formats used in cryo-ET.

Supported formats:
- IMOD
- AreTomo3
- Warp (global tilt-series alignment)
- RELION 5 (global tilt-series alignment)
- cryoet-data-portal

# Installation

`cryoet-alignment` can be installed using pip:

```bash
pip install cryoet-alignment
```

# Usage

## Reading and writing alignment files

`cryoet-alignment` provides a simple API to read and write alignment files from different software packages.

### IMOD
When processing tomography data using IMOD/etomo, files containing relevant alignment information are usually stored
following the naming convention `basename.xf` (in-plane parameters), `basename.tlt` (tilt angles),
`basename.xtilt` (x-rotation), `basename.mrc` (unaligned tilt series) and `basename_full_rec.mrc` (tomogram).

This layout is assumed when reading and writing IMOD alignment files as shown below. Any present `tilt.com` and
`newst.com` file in the same directory may also be read.

```python
from cryoet_alignment import read
from cryoet_alignment import write

# Read IMOD alignment files using etomo basename
imod_alignment = read("/path/to/imod_dir/basename")

# Write IMOD file
write(imod_alignment, "/path/to/imod_dir/basename")
```

### AreTomo3
When processing tomography data using AreTomo3, alignment information is stored in a single `.aln` file. This file can be
read and written as shown below.

```python
from cryoet_alignment import read
from cryoet_alignment import write

# Read AreTomo3 alignment files
aretomo3_alignment = read("/path/to/alignment_file.aln")

# Write AreTomo3 file
write(aretomo3_alignment, "/path/to/alignment_file.aln")
```

AreTomo3's per-tilt CTF estimates (`<name>_CTF.txt`, one row per raw tilt including dark frames) can also be read and
written. Because the `.txt` extension is generic, the reader is never inferred — pass `reader="aretomo3_ctf"` explicitly.

```python
from cryoet_alignment import read
from cryoet_alignment import write

# Read AreTomo3 CTF estimates
aretomo3_ctf = read("/path/to/name_CTF.txt", reader="aretomo3_ctf")

# Write AreTomo3 CTF file
write(aretomo3_ctf, "/path/to/name_CTF.txt")
```

AreTomo3's per-tilt angle file (`<name>_TLT.txt`: tilt angle, 1-based acquisition index and per-image dose, one row per
raw tilt including dark frames; AreTomo3 reads it in preference to `<name>.rawtlt`) is supported the same way. One- and
two-column files are accepted; pass `reader="aretomo3_tlt"` explicitly.

```python
from cryoet_alignment import read, write

# Read AreTomo3 tilt file
aretomo3_tlt = read("/path/to/name_TLT.txt", reader="aretomo3_tlt")
aretomo3_tlt.tilts, aretomo3_tlt.acq_indices, aretomo3_tlt.doses

# Write AreTomo3 tilt file
write(aretomo3_tlt, "/path/to/name_TLT.txt")
```

### Warp
Warp stores tilt-series metadata (including the global alignment: tilt angles, per-tilt tilt-axis rotation, and 2D
shifts in Å) in an XML file per tilt series. Only the global alignment is modeled; Warp's local warp grids have no
analog in the other formats and are ignored. Because the `.xml` extension is generic, the reader is never inferred —
pass `reader="warp"` explicitly. The tilt-image pixel size (Å/px) is needed to convert the Å shifts to pixels; if not
given, it is read from the XML's CTF `PixelSize` parameter.

```python
from cryoet_alignment import read
from cryoet_alignment import write

# Read a Warp tilt-series XML
warp_alignment = read("/path/to/tilt_series.xml", reader="warp", pixel_size_a=1.7005)

# Write a Warp tilt-series XML
write(warp_alignment, "/path/to/tilt_series.xml")
```

Two ancillary Warp project files are supported as well: the `.tomostar` tilt-series descriptor (`_wrpMovieName`,
`_wrpAngleTilt`, `_wrpAxisAngle`, `_wrpDose`, extra columns preserved) and the `.settings` project file
(`WarpTools create_settings`; `WarpSettings.create(...)` fills the vendored default document exactly like
`create_settings` does). Both are inferred from their extensions.

```python
from cryoet_alignment import read, write
from cryoet_alignment.io.warp import WarpSettings, WarpTomostar

tomostar = read("/path/to/tomostar/TS_01.tomostar")
tomostar.movie_names, [r.angle_tilt for r in tomostar.rows]

settings = WarpSettings.create(pixel_size_a=1.54, exposure_per_tilt=3.87, tomo_dims_px=(4096, 4096, 2000))
write(settings, "/path/to/warp_tiltseries.settings")
```

### RELION 5
RELION 5 stores tilt-series metadata in a `tomograms.star` plus per-tomogram star files. Both on-disk layouts are read:
the RELION-5 layout (`rlnTomoTiltSeriesStarFile` references) and the relion-4/WarpTools layout (per-tomogram blocks
embedded in one file), including relion-4 projection matrices (`rlnTomoProjX/Y/Z/W`), which are decomposed into Euler
angles and shifts exactly like RELION does it. Only the global alignment is modeled. Writing emits the RELION-5
two-file layout (`tomograms.star` + `tilt_series/<name>.star`). Pass `tomo_name` when the file lists several tomograms,
and `image_size_px` when the per-tilt table carries matrices instead of Euler columns.

```python
from cryoet_alignment import read
from cryoet_alignment import write

# Read a RELION tomograms.star (one tomogram)
relion_alignment = read("/path/to/tomograms.star", tomo_name="TS_01")

# Write the RELION-5 two-file layout
write(relion_alignment, "/path/to/out/tomograms.star")
```

### cryoet-data-portal
Alignment information from the cryoet-data-portal is stored in a JSON file with a schema described here. This file can
be read and written as shown below.

```python
from cryoet_alignment import read
from cryoet_alignment import write

# Read cryoet-data-portal alignment files
cryoet_data_portal_alignment = read("/path/to/alignment_file.json")

# Write cryoet-data-portal file
write(cryoet_data_portal_alignment, "/path/to/alignment_file.json")
```

## Convert between different alignment formats

`cryoet-alignment` provides the ability to convert between different alignment formats. For any conversion, the
alignment object must be read first using the appropriate `read` function, and then converted to the cryoet-data-portal
format before converting and writing to the desired format.

### IMOD to AreTomo3
```python
from cryoet_alignment import read, write
from cryoet_alignment.io.cryoet_data_portal import Alignment

# Read IMOD alignment files using etomo basename
imod_alignment = read("/path/to/imod_dir/basename")

# Convert IMOD to AreTomo3
cdp_alignment = Alignment.from_imod(imod_alignment)

# Write AreTomo3 file
tilt_series_dim = (4096, 4096, 41)
write(cdp_alignment.to_aretomo(ts_size=tilt_series_dim), "/path/to/alignment_file.aln")
```

### AreTomo3 to Warp
```python
from cryoet_alignment import read, write
from cryoet_alignment.io.cryoet_data_portal import Alignment

# Read the AreTomo3 alignment and convert to the canonical format
cdp_alignment = Alignment.from_aretomo3(read("/path/to/alignment_file.aln"), vol_size=(4096, 4096, 2000))

# Convert to Warp and write the tilt-series XML
warp_alignment = cdp_alignment.to_warp(pixel_size_a=1.54, image_size_px=(4096, 4096))
write(warp_alignment, "/path/to/tilt_series.xml")
```

### AreTomo3 to RELION
```python
from cryoet_alignment import read, write
from cryoet_alignment.io.cryoet_data_portal import Alignment

# Read the AreTomo3 alignment and convert to the canonical format
cdp_alignment = Alignment.from_aretomo3(read("/path/to/alignment_file.aln"), vol_size=(4096, 4096, 2000))

# Convert to RELION and write tomograms.star + tilt_series/TS_01.star
relion_alignment = cdp_alignment.to_relion("TS_01", pixel_size_a=1.54)
write(relion_alignment, "/path/to/out/tomograms.star")
```

### cryoet-data-portal to IMOD

It is also possible to convert directly from the cryoet-data-portal client to IMOD/AreTomo format. This is demonstrated
below. This requires additional dependencies to be installed using the following command:

```bash
pip install cryoet-alignment[cdp]
```

To convert from the cryoet-data-portal to IMOD, the below code can be used. Briefly, given a tomogram ID, the snippet
fetches the alignment information from the cryoet-data-portal, reads the tilt series metadata, and converts the alignment
to IMOD format. The resulting alignment files are written to the specified directory with the portal's run name as
the base name.

```python
import cryoet_data_portal as cdp
import zarr
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment import write

# Target tomogram ID
# This is an example from dataset 10004 (https://cryoetdataportal.czscience.com/runs/333)
TOMO_ID = 771

# Get the tomogram from the cryoet-data-portal
client = cdp.Client()
tomogram = cdp.Tomogram.get_by_id(client, TOMO_ID)

# Read cryoet-data-portal alignment from S3
cdp_ali = Alignment.from_s3(tomogram.alignment.s3_alignment_metadata)

# Get the tilt series metadata
#tilt_series = tomogram.alignment.tiltseries < currently unavailable due to a bug in the data portal client
tilt_series = tomogram.run.tiltseries[0]
pixel_size = tilt_series.pixel_spacing
dim_z, dim_y, dim_x = zarr.open(tilt_series.s3_omezarr_dir)['0'].shape

# Convert to IMOD format
imod_ali = cdp_ali.to_imod(ts_size=(dim_x, dim_y, dim_z), ts_spacing=pixel_size)
write(imod_ali, f"/tmp/test/{tomogram.run.name}")
```
