"""AreTomo3 ``<name>_TLT.txt`` per-tilt angle / acquisition-order / dose file.

AreTomo3 writes one line per section of the tilt-angle-sorted raw stack
(dark frames included) as ``"%8.2f  %4d  %8.2f"`` (DataUtil/CTsPackage.cpp,
``mSaveTiltFile``):

* col 1  tilt angle [degrees] (stage angle as acquired; ``.aln`` TILT values
  additionally carry ``AlphaOffset``)
* col 2  acquisition index, 1-based (the writer converts 0-based indices)
* col 3  dose received by this image [e/A^2] (the mdoc ``ExposureDose`` when
  ``-TotalDose 0``)

On input AreTomo3 (``mLoadTiltFile``) tries ``<name>_TLT.txt`` BEFORE
``<name>.rawtlt`` and parses each line with ``sscanf("%f %d %f")``, accepting
one, two or three columns and reading exactly ``nz`` lines. This model
mirrors that: ``acq_index`` and ``dose`` are optional, the column count must
be uniform across the file, acquisition indices are normalized to 1-based on
construction, and serialization is byte-identical to AreTomo3's own writer
for three-column files.
"""

import math
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from cryoet_alignment.io.base import FileIOBase


class TltInfo(BaseModel):
    """One ``_TLT.txt`` row.

    Attributes:
        tilt (float): Tilt angle in degrees.
        acq_index (int): 1-based acquisition index; None for one-column files.
        dose (float): Dose received by this image in e/A^2; None when absent.
    """

    model_config = ConfigDict(extra="forbid")

    tilt: float
    acq_index: Optional[int] = None
    dose: Optional[float] = None


class AreTomo3TLT(FileIOBase):
    """AreTomo3 ``_TLT.txt`` file: per-raw-section tilt angles, acquisition
    order and dose (dark frames included).

    Attributes:
        rows (List[TltInfo]): Rows in file (raw-stack) order.
    """

    model_config = ConfigDict(extra="forbid")

    rows: List[TltInfo]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def tilts(self) -> List[float]:
        return [r.tilt for r in self.rows]

    @property
    def has_acq_index(self) -> bool:
        return self.rows[0].acq_index is not None

    @property
    def has_dose(self) -> bool:
        return self.rows[0].dose is not None

    @property
    def acq_indices(self) -> Optional[List[int]]:
        """1-based acquisition indices, or None for one-column files."""
        if not self.has_acq_index:
            return None
        return [r.acq_index for r in self.rows]

    @property
    def doses(self) -> Optional[List[float]]:
        if not self.has_dose:
            return None
        return [r.dose for r in self.rows]

    def model_post_init(self, context, /) -> None:
        if not self.rows:
            raise ValueError("_TLT.txt contains no data rows")
        n_cols = {(r.acq_index is not None) + (r.dose is not None) for r in self.rows}
        if len(n_cols) != 1:
            raise ValueError("_TLT.txt rows do not all have the same number of columns")
        if any(r.dose is not None and r.acq_index is None for r in self.rows):
            raise ValueError("_TLT.txt rows carry a dose without an acquisition index")
        for r in self.rows:
            if not math.isfinite(r.tilt):
                raise ValueError("non-finite tilt angle in _TLT.txt")
            if r.dose is not None and (not math.isfinite(r.dose) or r.dose < 0):
                raise ValueError(f"dose must be finite and non-negative, got {r.dose}")
        if self.has_acq_index:
            acq = [r.acq_index for r in self.rows]
            base = min(acq)
            if base not in (0, 1):
                raise ValueError(f"acquisition indices must start at 0 or 1, got minimum {base}")
            if sorted(a - base for a in acq) != list(range(len(acq))):
                raise ValueError(f"acquisition indices are not a permutation of 1..{len(acq)}: {acq}")
            if base == 0:
                # AreTomo3's writer normalizes 0-based indices to 1-based (CTsPackage.cpp mSaveTiltFile)
                object.__setattr__(
                    self,
                    "rows",
                    [TltInfo(tilt=r.tilt, acq_index=r.acq_index + 1, dose=r.dose) for r in self.rows],
                )

    @classmethod
    def from_string(cls, text: str) -> "AreTomo3TLT":
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            tilt = float(parts[0])
            acq = int(parts[1]) if len(parts) >= 2 else None
            dose = float(parts[2]) if len(parts) >= 3 else None
            rows.append(TltInfo(tilt=tilt, acq_index=acq, dose=dose))
        return cls(rows=rows)

    def __str__(self) -> str:
        out = []
        for r in self.rows:
            if r.dose is not None:
                out.append(f"{r.tilt:8.2f}  {r.acq_index:4d}  {r.dose:8.2f}")
            elif r.acq_index is not None:
                out.append(f"{r.tilt:8.2f}  {r.acq_index:4d}")
            else:
                out.append(f"{r.tilt:8.2f}")
        return "\n".join(out) + "\n"
