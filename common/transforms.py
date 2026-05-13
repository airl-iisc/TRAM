"""Custom image transforms used across TRAM variants.

The CLAHE transform is the only non-trivial one — it is what makes V3
different from V2 at the input side.
"""

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    """Contrast Limited Adaptive Histogram Equalization on the L channel of LAB.

    For RGB inputs we operate only on luminance (L) so we don't shift colour.
    For grayscale we apply CLAHE directly. Returns a PIL.Image to slot into a
    torchvision Compose pipeline.
    """

    def __init__(self, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, img):
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_np = self.clahe.apply(img_np)
        return Image.fromarray(img_np)
