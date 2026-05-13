"""Run TRAM V3 inference on a folder of images.

CLAHE is applied via the transform pipeline so the same `SSSDataset` works.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from final.common.coco_eval import evaluate
from final.common.dataset import (
    DEFAULT_LABEL_MAP,
    SSSDataset,
    collate_fn,
    default_transform,
)
from final.common.transforms import CLAHETransform
from final.common.visualization import draw_boxes, overlay_masks, unnormalize
from final.tram_v3.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRAM V3 inference")
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="runs/tram_v3/inference")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-grid", type=int, default=8)
    p.add_argument("--evaluate-map", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detection_dir = os.path.join(args.output_dir, "detection")
    segmentation_dir = os.path.join(args.output_dir, "segmentation")
    os.makedirs(detection_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = DEFAULT_LABEL_MAP

    clahe = CLAHETransform(clipLimit=args.clahe_clip,
                           tileGridSize=(args.clahe_grid, args.clahe_grid))
    transform = default_transform(extra_first=clahe)

    dataset = SSSDataset(
        os.path.join(args.data_root, "images"),
        os.path.join(args.data_root, "labels"),
        transform=transform,
        label_map=label_map,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = build_model(num_classes=len(label_map) + 1, pretrained=False)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    if args.evaluate_map:
        metrics = evaluate(model, loader, device, label_map)
        print("=== Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print(f"Saving visualizations to {args.output_dir} ...")
    for batch in loader:
        images, targets = batch
        image_tensor = images[0]
        target = targets[0]
        with torch.no_grad():
            output = model([image_tensor.to(device)])[0]

        gt_classes = [label_map.get(l, str(l)) for l in target["labels"].cpu().tolist()]
        prefix = f"{int(target['image_id'].item())}_{'_'.join(sorted(set(gt_classes)))}"
        pil_image = unnormalize(image_tensor)

        boxes = output["boxes"].cpu().tolist()
        labels = output["labels"].cpu().tolist()
        scores = output["scores"].cpu().tolist()
        masks = output["masks"].cpu().numpy()

        keep = [i for i, s in enumerate(scores) if s >= args.conf_threshold]
        f_boxes = [boxes[i] for i in keep]
        f_labels = [labels[i] for i in keep]
        f_scores = [scores[i] for i in keep]
        f_masks = [(masks[i, 0] > 0.5).astype(np.uint8) for i in keep]

        gt_image = draw_boxes(pil_image.copy(), target["boxes"].cpu().tolist(),
                              target["labels"].cpu().tolist(),
                              box_color="green", text_color="green", label_map=label_map)
        pred_image = draw_boxes(pil_image.copy(), f_boxes, f_labels, scores=f_scores,
                                box_color="red", text_color="red", label_map=label_map)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(gt_image); axs[0].set_title("Ground Truth"); axs[0].axis("off")
        axs[1].imshow(pred_image); axs[1].set_title("Prediction"); axs[1].axis("off")
        plt.savefig(os.path.join(detection_dir, f"{prefix}_detection.png"), bbox_inches="tight")
        plt.close(fig)

        gt_seg = overlay_masks(pil_image.copy(),
                               [m.cpu().numpy() for m in target["masks"]],
                               color=(0, 255, 0))
        pred_seg = overlay_masks(pil_image.copy(), f_masks, color=(255, 0, 0))
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(gt_seg); axs[0].set_title("GT Mask"); axs[0].axis("off")
        axs[1].imshow(pred_seg); axs[1].set_title("Predicted Mask"); axs[1].axis("off")
        plt.savefig(os.path.join(segmentation_dir, f"{prefix}_segmentation.png"), bbox_inches="tight")
        plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
