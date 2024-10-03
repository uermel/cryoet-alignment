import os
from typing import Union

from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.imod import ImodAlignment

PATH_TYPE = Union[str, bytes, os.PathLike]


def read_imod(
    xf_path: PATH_TYPE,
    tlt_path: PATH_TYPE,
    xtilt_path: PATH_TYPE = None,
    tiltcom_path: PATH_TYPE = None,
    newstcom_path: PATH_TYPE = None,
) -> ImodAlignment:
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
    return ImodAlignment.read(base_name=base_name)


def read_aretomo3(aln_path: PATH_TYPE) -> AreTomo3ALN:
    return AreTomo3ALN.from_file(aln_path)


def read_cdp(cdp_path: PATH_TYPE) -> Alignment:
    return Alignment.from_cdp(cdp_path)


READER = {
    "imod": read_imod_basename,
    "aretomo3": read_aretomo3,
    "cdp": read_cdp,
}

INFER_READER = {
    ".aln": "aretomo3",
    ".json": "cdp",
}


def read(path: PATH_TYPE, reader: str = None) -> Union[AreTomo3ALN, Alignment, ImodAlignment]:
    if reader is None:
        _, ext = os.path.splitext(path)
        reader = INFER_READER.get(ext, "imod")

    return READER[reader](path)
