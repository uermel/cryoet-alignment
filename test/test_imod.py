from pathlib import Path
from typing import Tuple

from cryoet_alignment.io.imod.newst import ImodNEWSTCOM
from cryoet_alignment.io.imod.rawtlt import ImodRAWTLT
from cryoet_alignment.io.imod.tilt import ImodTILTCOM
from cryoet_alignment.io.imod.xf import ImodXF


def test_newst(newstcom_file: Tuple[Path, ImodNEWSTCOM]):
    path, exp = newstcom_file

    newst = ImodNEWSTCOM.from_file(path)

    for key in exp.model_fields:
        assert getattr(newst, key) == getattr(exp, key), f"Field {key} does not match."

    with open(path, "r") as f:
        assert str(newst) == f.read(), "Serialization does not match."


def test_rawtlt(rawtlt_file: Tuple[Path, ImodRAWTLT]):
    path, exp = rawtlt_file

    rawtlt = ImodRAWTLT.from_file(path)

    for key in exp.model_fields:
        assert getattr(rawtlt, key) == getattr(exp, key), f"Field {key} does not match."

    with open(path, "r") as f:
        assert str(rawtlt) == f.read(), "Serialization does not match."


def test_tilt(tiltcom_file: Tuple[Path, ImodTILTCOM]):
    path, exp = tiltcom_file

    tilt = ImodTILTCOM.from_file(path)

    for key in exp.model_fields:
        assert getattr(tilt, key) == getattr(exp, key), f"Field {key} does not match."

    with open(path, "r") as f:
        assert str(tilt) == f.read(), "Serialization does not match."


def test_xf(xf_file: Tuple[Path, ImodXF]):
    path, exp = xf_file

    xf = ImodXF.from_file(path)

    for key in exp.model_fields:
        assert getattr(xf, key) == getattr(exp, key), f"Field {key} does not match."

    with open(path, "r") as f:
        assert f.read() == str(xf), "Serialization does not match."
