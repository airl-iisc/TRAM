# TRAM V1 — MST + FPN + Mask R-CNN

The base TRAM variant from the paper. A Modified Swin Transformer (MST)
backbone produces four hierarchical feature maps; a Feature Pyramid Network
fuses them and feeds Mask R-CNN's RPN, ROI Align, box head, and mask head.

## Architecture

```
Input (224 x 224, RGB)
  -> MST backbone (Swin-Base)
        Stage 1: 56 x 56 x 128
        Stage 2: 28 x 28 x 256
        Stage 3: 14 x 14 x 512
        Stage 4:  7 x  7 x 1024
  -> FPN (out channels = 256)
  -> Mask R-CNN
        RPN -> ROI Align -> {Box head, Mask head}
  -> Detection + Segmentation
```

Implemented in [`model.py`](model.py). The Swin backbone is loaded via
`timm.create_model("swin_base_patch4_window7_224", pretrained=True)`.

## Training

```bash
python -m final.tram_v1.train \
    --data-root /path/to/SSS_OD-5 \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --output-dir runs/tram_v1
```

Saves the best checkpoint (highest bbox mAP@0.5:0.95 on the validation split)
to `runs/tram_v1/tram_v1_best.pth` and writes the training log alongside.

## Inference

```bash
python -m final.tram_v1.inference \
    --weights runs/tram_v1/tram_v1_best.pth \
    --data-root /path/to/SSS_OD-5/valid \
    --output-dir runs/tram_v1/inference \
    --evaluate-map
```

Produces side-by-side ground-truth vs prediction images in
`runs/tram_v1/inference/detection/` and `…/segmentation/`. With
`--evaluate-map` it also reports COCO bbox and segm mAP.

## Reported numbers (paper)

| Metric                | Value  |
|-----------------------|--------|
| Detection mAP@0.5     | 0.8657 |
| Detection mAP@0.5:0.95| 0.5391 |
| Segmentation mAP@0.5  | 0.7226 |
| Segmentation mAP@0.5:0.95 | 0.4098 |
| Epochs to converge    | 100    |
