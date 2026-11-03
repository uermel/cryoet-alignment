import os
from typing import Union

from cryoet_alignment.io.aretomo3 import AreTomo3ALN, AreTomo3CTF, AreTomo3TLT
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.imod import ImodAlignment
from cryoet_alignment.io.relion import RelionAlignment
from cryoet_alignment.io.warp import WarpAlignment

PATH_TYPE = Union[str, bytes, os.PathLike]


def read_imod(
    xf_path: PATH_TYPE,
    tlt_path: PATH_TYPE,
    xtilt_path: PATH_TYPE = None,
    tiltcom_path: PATH_TYPE = None,
    newstcom_path: PATH_TYPE = None,
) -> ImodAlignment:
    """Read an IMOD alignment from the specified files.

    Args:
        xf_path: The path to the .xf file.
        tlt_path: The path to the .tlt file.
        xtilt_path: The path to the .xtilt file.
        tiltcom_path: The path to the .tiltcom file.
        newstcom_path: The path to the .newstcom file.

    Returns:
        ImodAlignment: The alignment object.
    """
    return ImodAlignment.read(
        xf_path=xf_path,
        tlt_path=tlt_path,
        xtilt_path=xtilt_path,
        tiltcom_path=tiltcom_path,
        newstcom_path=newstcom_path,
    )


def read_imod_basename(
    base_name: str,
) -> ImodAlignment:
    """Read an IMOD alignment from the specified basename. The alignment files should be named `{base_name}.xf`,
    `{base_name}.tlt`, `{base_name}.xtilt`. `tilt.com` and `newst.com` files in the same directory may also be read.

    Args:
        base_name: The basename of the alignment files.

    Returns:
        ImodAlignment: The alignment object.
    """
    return ImodAlignment.read(base_name=base_name)


def read_aretomo3(aln_path: PATH_TYPE) -> AreTomo3ALN:
    """Read an AreTomo3 alignment from the specified file.

    Args:
        aln_path: The path to the .aln file.

    Returns:
        AreTomo3ALN: The alignment object.
    """
    return AreTomo3ALN.from_file(aln_path)


def read_cdp(cdp_path: PATH_TYPE) -> Alignment:
    """Read a CryoET Data Portal alignment from the specified file.

    Args:
        cdp_path: The path to the .json file.

    Returns:
        Alignment: The alignment object.
    """
    return Alignment.from_file(cdp_path)


def read_aretomo3_ctf(ctf_path: PATH_TYPE) -> AreTomo3CTF:
    """Read AreTomo3 per-tilt CTF estimates from the specified _CTF.txt file.

    Args:
        ctf_path: The path to the _CTF.txt file.

    Returns:
        AreTomo3CTF: The CTF object.
    """
    return AreTomo3CTF.from_file(ctf_path)


def read_aretomo3_tlt(tlt_path: PATH_TYPE) -> AreTomo3TLT:
    """Read AreTomo3 per-tilt angles / acquisition order / dose from a _TLT.txt file.

    Args:
        tlt_path: The path to the _TLT.txt file.

    Returns:
        AreTomo3TLT: The tilt-file object.
    """
    return AreTomo3TLT.from_file(tlt_path)


def read_warp(xml_path: PATH_TYPE, pixel_size_a: float = None) -> WarpAlignment:
    """Read a Warp alignment from the specified tilt-series XML file.

    Args:
        xml_path: The path to the Warp tilt-series XML metadata file.
        pixel_size_a: Tilt-image pixel size in Å/px. If None, it is read from
            the XML's CTF PixelSize parameter; an explicit value always wins.

    Returns:
        WarpAlignment: The alignment object.
    """
    return WarpAlignment.from_file(xml_path, pixel_size_a=pixel_size_a)


def read_relion(
    tomograms_star: PATH_TYPE,
    tomo_name: str = None,
    image_size_px: tuple = None,
) -> RelionAlignment:
    """Read a RELION 5 tilt-series alignment from a tomograms.star.

    Args:
        tomograms_star: The path to the tomograms.star (RELION-5 layout with
            rlnTomoTiltSeriesStarFile references, or the relion-4/WarpTools
            layout with embedded per-tomogram blocks).
        tomo_name: Which tomogram to load when the file lists several.
        image_size_px: Tilt-image (nx, ny) in px — required only when the
            per-tilt table carries projection matrices instead of Euler columns.

    Returns:
        RelionAlignment: The alignment object.
    """
    return RelionAlignment.from_file(tomograms_star, tomo_name=tomo_name, image_size_px=image_size_px)


READER = {
    "imod": read_imod_basename,
    "aretomo3": read_aretomo3,
    "aretomo3_ctf": read_aretomo3_ctf,
    "aretomo3_tlt": read_aretomo3_tlt,
    "cdp": read_cdp,
    "warp": read_warp,
    "relion": read_relion,
}

# Note: _CTF.txt and _TLT.txt are intentionally NOT auto-inferred — ``.txt`` is too
# generic. Pass ``reader="aretomo3_ctf"`` / ``reader="aretomo3_tlt"`` explicitly.
# Note: ``.xml`` is intentionally NOT auto-inferred — it is too generic and could
# collide with non-Warp XML formats. Pass ``reader="warp"`` explicitly.
# ``.star`` IS inferred: within this package's scope a .star file is a RELION
# tomograms.star.
INFER_READER = {
    ".aln": "aretomo3",
    ".json": "cdp",
    ".star": "relion",
}


def read(
    path: PATH_TYPE,
    reader: str = None,
    **kwargs,
) -> Union[AreTomo3ALN, AreTomo3CTF, AreTomo3TLT, Alignment, ImodAlignment, RelionAlignment, WarpAlignment]:
    """Read alignment files in IMOD, AreTomo3, CryoET Data Portal, RELION, or Warp format.

    Args:
        path: The path to the alignment file (or basename for IMOD).
        reader: The reader to use for the alignment file (one of "imod", "aretomo3",
        "aretomo3_ctf", "aretomo3_tlt", "cdp", "relion", or "warp"). If None, the reader
        will be inferred from the file extension (".aln", ".json", ".star").
        Note: ``.xml``, _CTF.txt and _TLT.txt are not auto-inferred — pass the reader explicitly.
        **kwargs: Passed through to the selected reader (e.g. ``pixel_size_a``
        for ``reader="warp"``; ``tomo_name``/``image_size_px`` for ``reader="relion"``).

    Returns:
        Union[AreTomo3ALN, AreTomo3CTF, AreTomo3TLT, Alignment, ImodAlignment, RelionAlignment, WarpAlignment]:
        The alignment object.
    """
    if reader is None:
        _, ext = os.path.splitext(path)
        reader = INFER_READER.get(ext, "imod")

    return READER[reader](path, **kwargs)
