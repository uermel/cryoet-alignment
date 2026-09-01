import os
from typing import Union

from cryoet_alignment.io.aretomo3 import AreTomo3ALN, AreTomo3CTF
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.imod import ImodAlignment
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
    return Alignment.from_cdp(cdp_path)


def read_aretomo3_ctf(ctf_path: PATH_TYPE) -> AreTomo3CTF:
    """Read AreTomo3 per-tilt CTF estimates from the specified _CTF.txt file.

    Args:
        ctf_path: The path to the _CTF.txt file.

    Returns:
        AreTomo3CTF: The CTF object.
    """
    return AreTomo3CTF.from_file(ctf_path)


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


READER = {
    "imod": read_imod_basename,
    "aretomo3": read_aretomo3,
    "aretomo3_ctf": read_aretomo3_ctf,
    "cdp": read_cdp,
    "warp": read_warp,
}

# Note: _CTF.txt is intentionally NOT auto-inferred — ``.txt`` is too generic.
# Pass ``reader="aretomo3_ctf"`` explicitly.
# Note: ``.xml`` is intentionally NOT auto-inferred — it is too generic and could
# collide with non-Warp XML formats. Pass ``reader="warp"`` explicitly.
INFER_READER = {
    ".aln": "aretomo3",
    ".json": "cdp",
}


def read(
    path: PATH_TYPE,
    reader: str = None,
    **kwargs,
) -> Union[AreTomo3ALN, AreTomo3CTF, Alignment, ImodAlignment, WarpAlignment]:
    """Read alignment files in IMOD, AreTomo3, CryoET Data Portal, or Warp format.

    Args:
        path: The path to the alignment file (or basename for IMOD).
        reader: The reader to use for the alignment file (one of "imod", "aretomo3",
        "aretomo3_ctf", "cdp", or "warp"). If None, the reader will be inferred from the file extension.
        Note: ``.xml`` and _CTF.txt are not auto-inferred — pass the reader explicitly.
        **kwargs: Passed through to the selected reader (e.g. ``pixel_size_a``
        for ``reader="warp"``).

    Returns:
        Union[AreTomo3ALN, AreTomo3CTF, Alignment, ImodAlignment, WarpAlignment]: The alignment object.
    """
    if reader is None:
        _, ext = os.path.splitext(path)
        reader = INFER_READER.get(ext, "imod")

    return READER[reader](path, **kwargs)
