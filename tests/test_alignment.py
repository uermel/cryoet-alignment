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


def test_from_aretomo3_legacy_permuted_sec():
    """z_index follows SEC, not row order, so a legacy .aln computed on an acquisition-ordered stack maps each
    tilt to its own section."""
    from pathlib import Path

    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    aln = AreTomo3ALN.from_file(Path(__file__).parent / "data" / "test_legacy_permuted.aln")
    ali = Alignment.from_aretomo3(aln=aln)

    by_z = {p.z_index: p.tilt_angle for p in ali.per_section_alignment_parameters}
    assert by_z == {0: 0.0, 1: 3.0, 2: -3.0, 3: 6.0, 4: -6.0, 5: 9.0, 6: -9.0}
