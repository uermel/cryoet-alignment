"""AreTomo3 ``<name>_CTF.txt`` per-tilt CTF estimation results.

Format (AreTomo3 CSaveCtfResults.cpp:69-99): header comment lines starting
with ``#``, then one row per RAW tilt (the tilt-angle-ascending stack
INCLUDING dark frames — CTF is estimated before dark removal,
CAreTomoMain.cpp:92-130), ``"%4d %8.2f %8.2f %8.2f %9.4f %8.4f %8.4f %3d"``:

* col 1  micrograph number (1-based row index into the sorted raw stack)
* col 2  defocus 1 = DfMax [Angstrom] (>= col 3 by construction upstream)
* col 3  defocus 2 = DfMin [Angstrom]
* col 4  azimuth of astigmatism [degrees] (direction of DfMax; NOT wrapped)
* col 5  additional phase shift [RADIANS]
* col 6  cross-correlation score
* col 7  fit resolution limit [Angstrom]
* col 8  dfHand (+1 or -1; the +1 normalization via a 180-degree tilt-axis
  rotation, CAreTomoMain.cpp:344-367, is CONDITIONAL on the -TiltAxis
  refine setting, so both values occur in practice)

CTFFIND4-style 7-column files (no dfHand column) are accepted, with
``df_hand = None``.

Row keying: the file carries ONLY the micrograph number — rows are joined to
other representations by normalizing the contiguous number column by its
minimum (AreTomo3's own loader accepts 0- or 1-based input the same way,
CLoadCtfResults.cpp:84-94) and treating the result as the ordinal into the
raw ascending-tilt-sorted stack including darks. Duplicates, gaps, and
non-contiguous numbering are rejected, and rows are stored in ordinal order
(``rows[i]`` is raw-stack ordinal ``i``). Serialization renumbers rows 1..N.
"""

import math
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from cryoet_alignment.io.base import FileIOBase

_HEADER = (
    "# Columns: #1 micrograph number; #2 - defocus 1 [A]; #3 - defocus 2; "
    "#4 - azimuth of astigmatism;\n"
    "#5 - additional phase shift [radian]; #6 - cross correlation;\n"
    "#7 - spacing (in Angstroms) up to which CTF rings were fit successfully; #8 - dfHand\n"
)


class CtfInfo(BaseModel):
    """One per-tilt CTF estimate (one ``_CTF.txt`` row).

    Attributes:
        micrograph (int): Micrograph number as read; the normalized ordinal is
            ``micrograph - min(column)``.
        df_max_a (float): Defocus 1 = DfMax in Angstrom.
        df_min_a (float): Defocus 2 = DfMin in Angstrom.
        azimuth_deg (float): Azimuth of astigmatism in degrees (direction of DfMax).
        phase_rad (float): Additional phase shift in RADIANS.
        score (float): Cross-correlation score.
        res_a (float): Fit resolution limit in Angstrom.
        df_hand (int): Defocus handedness (+1/-1); None for 7-column files.
    """

    model_config = ConfigDict(extra="forbid")

    micrograph: int
    df_max_a: float
    df_min_a: float
    azimuth_deg: float
    phase_rad: float
    score: float
    res_a: float
    df_hand: Optional[int] = None


class AreTomo3CTF(FileIOBase):
    """AreTomo3 ``_CTF.txt`` file: per-raw-tilt CTF estimates (darks included).

    Attributes:
        rows (List[CtfInfo]): Rows in raw-stack ordinal order (validated and
            re-sorted on construction).
    """

    model_config = ConfigDict(extra="forbid")

    rows: List[CtfInfo]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def model_post_init(self, context, /) -> None:
        if not self.rows:
            raise ValueError("_CTF.txt contains no data rows")
        nums = [r.micrograph for r in self.rows]
        base = min(nums)
        ordinals = [n - base for n in nums]
        if sorted(ordinals) != list(range(len(nums))):
            raise ValueError(
                "_CTF.txt micrograph-number column is not a contiguous unique "
                f"sequence (after normalizing by its minimum {base}): {nums}",
            )
        # store rows in ordinal order so rows[i] == raw-stack ordinal i
        order = sorted(range(len(nums)), key=lambda i: ordinals[i])
        object.__setattr__(self, "rows", [self.rows[i] for i in order])

        for r in self.rows:
            for name in ("df_max_a", "df_min_a", "azimuth_deg", "phase_rad", "score", "res_a"):
                if not math.isfinite(getattr(r, name)):
                    raise ValueError(f"non-finite {name} in _CTF.txt row {r.micrograph}")

    @classmethod
    def from_string(cls, text: str) -> "AreTomo3CTF":
        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) not in (7, 8):
                raise ValueError(f"malformed _CTF.txt line (expected 7 or 8 columns): {line!r}")
            rows.append(
                CtfInfo(
                    micrograph=int(parts[0]),
                    df_max_a=float(parts[1]),
                    df_min_a=float(parts[2]),
                    azimuth_deg=float(parts[3]),
                    phase_rad=float(parts[4]),
                    score=float(parts[5]),
                    res_a=float(parts[6]),
                    df_hand=int(parts[7]) if len(parts) == 8 else None,
                ),
            )
        return cls(rows=rows)

    def __str__(self) -> str:
        out = [_HEADER.rstrip("\n")]
        for i, r in enumerate(self.rows):
            hand = r.df_hand if r.df_hand is not None else 1
            out.append(
                f"{i + 1:4d} {r.df_max_a:8.2f} {r.df_min_a:8.2f} {r.azimuth_deg:8.2f} "
                f"{r.phase_rad:9.4f} {r.score:8.4f} {r.res_a:8.4f} {hand:3d}",
            )
        return "\n".join(out) + "\n"
