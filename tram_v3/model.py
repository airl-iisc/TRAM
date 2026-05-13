"""TRAM V3: V2 architecture (MST + FPN + CBAM + Mask R-CNN).

V3 differs from V2 only at the input side — CLAHE is added to the
preprocessing transform. The network itself is identical to V2, so we reuse
its `build_model`.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from final.tram_v2.model import build_model  # noqa: F401  (re-export)


__all__ = ["build_model"]
