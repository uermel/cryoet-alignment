from typing import Tuple

from mrcfile.mrcinterpreter import MrcInterpreter


def get_ts_size_local(file: str) -> Tuple[int, int, int]:
    with open(file, "rb") as f:
        mrc = MrcInterpreter(f, header_only=True)
        return mrc.header.nx, mrc.header.ny, mrc.header.nz


def get_mrc_header_local(file: str):
    with open(file, "rb") as f:
        mrc = MrcInterpreter(f, header_only=True)
        return mrc.header
