"""Visualization helpers for inference output.

Draws ground-truth and predicted boxes / masks side-by-side.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(img_tensor: torch.Tensor) -> Image.Image:
    img = img_tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    img = img.numpy().transpose(1, 2, 0)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img)


def draw_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    box_color: str = "red",
    text_color: str = "red",
    label_map: Optional[dict] = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        text = label_map.get(labels[i], str(labels[i])) if label_map else str(labels[i])
        if scores is not None:
            text += f": {scores[i]:.2f}"
        draw.text((x1, y1), text, fill=text_color)
    return image


def overlay_masks(
    image: Image.Image,
    masks: List[np.ndarray],
    color=(0, 255, 0),
    alpha: float = 0.4,
) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for mask in masks:
        colored = Image.new("RGBA", image.size, color + (int(255 * alpha),))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        overlay = Image.composite(colored, overlay, mask_img)
    return Image.alpha_composite(image, overlay).convert("RGB")
