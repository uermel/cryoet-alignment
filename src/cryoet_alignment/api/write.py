import os
from typing import Union

from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.imod import ImodAlignment

PATH_TYPE = Union[str, bytes, os.PathLike]


def write_imod(
    alignment: ImodAlignment,
    xf_path: PATH_TYPE,
    tlt_path: PATH_TYPE,
    xtilt_path: PATH_TYPE = None,
    tiltcom_path: PATH_TYPE = None,
    newstcom_path: PATH_TYPE = None,
) -> None:
    """Write an IMOD alignment to the specified files.

    Args:
        alignment: The alignment object to write.
        xf_path: The path to the .xf file.
        tlt_path: The path to the .tlt file.
        xtilt_path: The path to the .xtilt file.
        tiltcom_path: The path to the .tiltcom file.
        newstcom_path: The path to the .newstcom file.
    """

    alignment.write(
        xf_path=xf_path,
        tlt_path=tlt_path,
        xtilt_path=xtilt_path,
        tiltcom_path=tiltcom_path,
        newstcom_path=newstcom_path,
    )


def write_imod_basename(
    alignment: ImodAlignment,
    base_name: str,
) -> None:
    """Write an IMOD alignment to the specified basename. The alignment files will be written as `{base_name}.xf`,
    `{base_name}.tlt`, `{base_name}.xtilt`, `tilt.com` and `newst.com`.

    Args:
        alignment: The alignment object to write.
        base_name: The basename of the alignment files
    """
    alignment.write(base_name=base_name)


def write_aretomo3(aln: AreTomo3ALN, aln_path: PATH_TYPE) -> None:
    """Write an alignment in AreTomo3 format.

    Args:
        aln: The alignment object to write.
        aln_path: The path to write the alignment file to.
    """
    with open(aln_path, "w") as f:
        f.write(str(aln))


def write_cdp(ali: Alignment, cdp_path: PATH_TYPE) -> None:
    """Write an alignment in CryoET Data Portal format.

    Args:
        ali: The alignment object to write.
        cdp_path: The path to write the alignment file to.
    """
    with open(cdp_path, "w") as f:
        f.write(str(ali))


WRITER = {
    "imod": write_imod_basename,
    "aretomo3": write_aretomo3,
    "cdp": write_cdp,
}

INFER_WRITER = {
    Alignment: "cdp",
    AreTomo3ALN: "aretomo3",
    ImodAlignment: "imod",
}


def write(alignment: Union[Alignment, AreTomo3ALN, ImodAlignment], path: PATH_TYPE, writer: str = None) -> None:
    """Write alignment files in IMOD, AreTomo3, or CryoET Data Portal format.

    Args:
        alignment: The alignment object to write.
        path: The path to write the alignment file to.
        writer: The writer to use for the alignment file (one of "imod", "aretomo3" or "cdp"). If None, the writer will
        be inferred from the alignment object type.
    """
    if writer is None:
        writer = INFER_WRITER[type(alignment)]

    WRITER[writer](alignment, path)
