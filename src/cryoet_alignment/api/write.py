import os
from typing import Union

from cryoet_alignment.io.aretomo3 import AreTomo3ALN, AreTomo3CTF, AreTomo3TLT
from cryoet_alignment.io.cryoet_data_portal import Alignment
from cryoet_alignment.io.imod import ImodAlignment
from cryoet_alignment.io.relion import RelionAlignment
from cryoet_alignment.io.warp import WarpAlignment, WarpSettings, WarpTomostar

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


def write_aretomo3_ctf(ctf: AreTomo3CTF, ctf_path: PATH_TYPE) -> None:
    """Write per-tilt CTF estimates in AreTomo3 _CTF.txt format.

    Args:
        ctf: The CTF object to write.
        ctf_path: The path to write the _CTF.txt file to.
    """
    ctf.to_file(ctf_path)


def write_aretomo3_tlt(tlt: AreTomo3TLT, tlt_path: PATH_TYPE) -> None:
    """Write per-tilt angles / acquisition order / dose in AreTomo3 _TLT.txt format.

    Args:
        tlt: The tilt-file object to write.
        tlt_path: The path to write the _TLT.txt file to.
    """
    tlt.to_file(tlt_path)


def write_warp(warp: WarpAlignment, xml_path: PATH_TYPE) -> None:
    """Write a Warp alignment to a tilt-series XML metadata file.

    Args:
        warp: The alignment object to write.
        xml_path: The path to write the alignment file to.
    """
    warp.to_file(xml_path)


def write_warp_tomostar(tomostar: WarpTomostar, tomostar_path: PATH_TYPE) -> None:
    """Write a Warp .tomostar tilt-series descriptor.

    Args:
        tomostar: The tomostar object to write.
        tomostar_path: The path to write the .tomostar file to.
    """
    tomostar.to_file(tomostar_path)


def write_warp_settings(settings: WarpSettings, settings_path: PATH_TYPE) -> None:
    """Write a Warp .settings project file.

    Args:
        settings: The settings object to write.
        settings_path: The path to write the .settings file to.
    """
    settings.to_file(settings_path)


def write_relion(relion: RelionAlignment, tomograms_star: PATH_TYPE) -> None:
    """Write a RELION 5 tilt-series alignment in the RELION-5 two-file layout.

    Args:
        relion: The alignment object to write.
        tomograms_star: The path to write the tomograms.star to; the per-tilt
            table is written to ``tilt_series/<tomo_name>.star`` next to it.
    """
    relion.to_file(tomograms_star)


WRITER = {
    "imod": write_imod_basename,
    "aretomo3": write_aretomo3,
    "aretomo3_ctf": write_aretomo3_ctf,
    "aretomo3_tlt": write_aretomo3_tlt,
    "cdp": write_cdp,
    "warp": write_warp,
    "warp_tomostar": write_warp_tomostar,
    "warp_settings": write_warp_settings,
    "relion": write_relion,
}

INFER_WRITER = {
    Alignment: "cdp",
    AreTomo3ALN: "aretomo3",
    AreTomo3CTF: "aretomo3_ctf",
    AreTomo3TLT: "aretomo3_tlt",
    ImodAlignment: "imod",
    WarpAlignment: "warp",
    WarpTomostar: "warp_tomostar",
    WarpSettings: "warp_settings",
    RelionAlignment: "relion",
}


def write(
    alignment: Union[
        Alignment,
        AreTomo3ALN,
        AreTomo3CTF,
        AreTomo3TLT,
        ImodAlignment,
        RelionAlignment,
        WarpAlignment,
        WarpTomostar,
        WarpSettings,
    ],
    path: PATH_TYPE,
    writer: str = None,
) -> None:
    """Write alignment files in IMOD, AreTomo3, CryoET Data Portal, RELION, or Warp format.

    Args:
        alignment: The alignment object to write.
        path: The path to write the alignment file to.
        writer: The writer to use for the alignment file (one of "imod", "aretomo3",
        "aretomo3_ctf", "aretomo3_tlt", "cdp", "relion", "warp", "warp_tomostar" or
        "warp_settings"). If None, the writer will be inferred from the alignment object type.
    """
    if writer is None:
        writer = INFER_WRITER[type(alignment)]

    WRITER[writer](alignment, path)
