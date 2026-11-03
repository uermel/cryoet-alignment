"""CETS (TomoBabel cryo-ET standard) rigid tilt-series profile ``cets-rigid/0.2``.

The ``cets_data_model`` package (git-only, pinned commit in ``docs/cets.md``) is imported lazily so the
rest of cryoet-alignment does not need it. Every public function here raises a clear ``ImportError``
pointing at the install line when it is missing.
"""

from cryoet_alignment.io.cets.profile import PROFILE_VERSION, require_cets

__all__ = ["PROFILE_VERSION", "require_cets"]
