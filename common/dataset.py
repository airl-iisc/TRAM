"""SSS_OD dataset loader.

Reads the SeabedObjects-KLSG-derived dataset in YOLOv5-polygon layout:
    images/<name>.{jpg,png}
    labels/<name>.txt        # "<class_idx> x1 y1 x2 y2 ... xn yn" normalized

Class indices in the label files are 0-based ({0: plane, 1: ship}).
We shift them by +1 internally so 0 is reserved for background, matching the
Mask R-CNN convention.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


DEFAULT_LABEL_MAP = {1: "plane", 2: "ship"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(image_size: int = 224, extra_first: Optional[Callable] = None) -> Compose:
    """Build the standard preprocessing pipeline.

    `extra_first` lets V3 inject CLAHE before resize.
    """
    steps = []
    if extra_first is not None:
        steps.append(extra_first)
    steps.extend([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return Compose(steps)


class SSSDataset(Dataset):
    """Side-Scan Sonar object dataset with polygon masks and bounding boxes."""

    def __init__(
        self,
        image_folder: str,
        bbox_folder: str,
        transform: Optional[Compose] = None,
        label_map: Optional[dict] = None,
        image_size: int = 224,
    ):
        self.image_folder = image_folder
        self.bbox_folder = bbox_folder
        self.transform = transform if transform is not None else default_transform(image_size)
        self.label_map = label_map if label_map is not None else DEFAULT_LABEL_MAP

        all_image_files = sorted(
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        self.image_files = []
        for img_file in all_image_files:
            base_name = os.path.splitext(img_file)[0]
            bbox_path = os.path.join(self.bbox_folder, f"{base_name}.txt")
            if not os.path.exists(bbox_path):
                continue
            polygons, labels = self._load_annotation(bbox_path)
            if labels:
                self.image_files.append(img_file)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        base_name = os.path.splitext(self.image_files[idx])[0]
        bbox_path = os.path.join(self.bbox_folder, f"{base_name}.txt")
        polygons, labels = self._load_annotation(bbox_path)
        if not labels:
            return None

        boxes = []
        masks = []
        for polygon_normalized in polygons:
            polygon = [(x * width, y * height) for x, y in polygon_normalized]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

            mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
            masks.append(mask)

        image = self.transform(image)
        target_h, target_w = image.shape[1], image.shape[2]

        masks_resized = []
        for mask in masks:
            mask_resized = mask.resize((target_w, target_h), resample=Image.NEAREST)
            masks_resized.append(torch.tensor(np.array(mask_resized), dtype=torch.uint8))
        masks_tensor = torch.stack(masks_resized)

        scale_x = target_w / width
        scale_y = target_h / height
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        boxes_tensor[:, [0, 2]] *= scale_x
        boxes_tensor[:, [1, 3]] *= scale_y

        # 0-based -> 1-based (0 is background for Mask R-CNN).
        labels_tensor = torch.tensor(labels, dtype=torch.int64) + 1

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
        }
        return image, target

    @staticmethod
    def _load_annotation(file_path: str):
        boxes, labels = [], []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3 or (len(parts) - 1) % 2 != 0:
                    continue
                label_idx = int(parts[0])
                coords = list(map(float, parts[1:]))
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                boxes.append(points)
                labels.append(label_idx)
        return boxes, labels


def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    return tuple(zip(*batch))


def mask_to_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def create_weighted_sampler(dataset: SSSDataset) -> WeightedRandomSampler:
    """Balance class frequency. The dataset is plane-poor relative to ships."""
    label_counts = Counter()
    for _, target in dataset:
        label_counts.update(target["labels"].tolist())
    total = len(dataset)
    class_weights = {label: total / count for label, count in label_counts.items()}
    samples_weight = []
    for _, target in dataset:
        labels = target["labels"].tolist()
        samples_weight.append(np.mean([class_weights[label] for label in labels]))
    return WeightedRandomSampler(
        samples_weight, num_samples=len(samples_weight), replacement=True
    )
