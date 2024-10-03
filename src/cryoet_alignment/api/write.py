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
    alignment.write(base_name=base_name)


def write_aretomo3(aln: AreTomo3ALN, aln_path: PATH_TYPE) -> None:
    with open(aln_path, "w") as f:
        f.write(str(aln))


def write_cdp(ali: Alignment, cdp_path: PATH_TYPE) -> None:
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
    if writer is None:
        writer = INFER_WRITER[type(alignment)]

    WRITER[writer](alignment, path)
