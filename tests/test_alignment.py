from cryoet_alignment.io.cryoet_data_portal.alignment import Alignment


def test_imod_consistency(imod_base: str):
    path = imod_base

    ali = Alignment.from_imod_basename(path)
    imod_alignment = ali.to_imod([3832, 3708, 59], 1)
    xf = imod_alignment.xf
    tlt = imod_alignment.tlt
    xtilt = imod_alignment.xtilt

    with open(f"{path}.xf", "r") as f:
        assert f.read() == str(xf), "XF serialization does not match."

    with open(f"{path}.tlt", "r") as f:
        assert f.read() == str(tlt), "TLT serialization does not match."

    with open(f"{path}.xtilt", "r") as f:
        assert f.read() == str(xtilt), "XTILT serialization does not match."
