"""COCO-style mAP evaluation for Mask R-CNN outputs.

Used by training and standalone validation. Reports both detection (bbox) and
segmentation (mask) mAP at IoU=0.5 and 0.5:0.95.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Dict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset import mask_to_rle


@torch.no_grad()
def evaluate(model, data_loader, device, label_map: Dict[int, str], silent: bool = False) -> Dict[str, float]:
    model.eval()
    coco_gt = COCO()
    coco_dt = COCO()
    coco_gt.dataset = {"images": [], "annotations": [], "categories": []}
    coco_dt.dataset = {"images": [], "annotations": [], "categories": []}
    coco_gt.dataset["categories"] = [{"id": idx, "name": name} for idx, name in label_map.items()]
    coco_dt.dataset["categories"] = [{"id": idx, "name": name} for idx, name in label_map.items()]
    ann_id = 1
    dt_ann_id = 1

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for idx in range(len(images)):
            image_id = int(targets[idx]["image_id"].item())
            height, width = images[idx].shape[1], images[idx].shape[2]
            coco_gt.dataset["images"].append({"id": image_id, "height": height, "width": width})
            coco_dt.dataset["images"].append({"id": image_id, "height": height, "width": width})

            gt_boxes = targets[idx]["boxes"].cpu().numpy()
            gt_labels = targets[idx]["labels"].cpu().numpy()
            gt_masks = targets[idx]["masks"].cpu().numpy()
            for j in range(len(gt_boxes)):
                xmin, ymin, xmax, ymax = gt_boxes[j]
                w_box, h_box = xmax - xmin, ymax - ymin
                coco_gt.dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(gt_labels[j]),
                    "bbox": [float(xmin), float(ymin), float(w_box), float(h_box)],
                    "area": float(w_box * h_box),
                    "iscrowd": 0,
                    "segmentation": mask_to_rle(gt_masks[j]),
                })
                ann_id += 1

            pred_boxes = outputs[idx]["boxes"].cpu().numpy()
            scores = outputs[idx]["scores"].cpu().numpy()
            pred_labels = outputs[idx]["labels"].cpu().numpy()
            pred_masks = outputs[idx]["masks"].cpu().numpy()
            for k in range(len(pred_boxes)):
                xmin, ymin, xmax, ymax = pred_boxes[k]
                w_box, h_box = xmax - xmin, ymax - ymin
                coco_dt.dataset["annotations"].append({
                    "id": dt_ann_id,
                    "image_id": image_id,
                    "category_id": int(pred_labels[k]),
                    "bbox": [float(xmin), float(ymin), float(w_box), float(h_box)],
                    "area": float(w_box * h_box),
                    "score": float(scores[k]),
                    "segmentation": mask_to_rle((pred_masks[k, 0] > 0.5).astype(np.uint8)),
                })
                dt_ann_id += 1

    coco_gt.createIndex()
    coco_dt.createIndex()

    sink = io.StringIO() if silent else None
    with redirect_stdout(sink) if silent else _no_redirect():
        bbox_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()
        mask_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        mask_eval.evaluate()
        mask_eval.accumulate()
        mask_eval.summarize()

    return {
        "mAP_50_95_bbox": float(bbox_eval.stats[0]),
        "mAP_50_bbox": float(bbox_eval.stats[1]),
        "mAP_50_95_mask": float(mask_eval.stats[0]),
        "mAP_50_mask": float(mask_eval.stats[1]),
    }


class _no_redirect:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
