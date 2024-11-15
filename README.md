# cryoet-alignment

Convert between different alignment formats used in cryo-ET.

Supported formats:
- IMOD
- AreTomo3
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
