# TRAM V3 — V2 + CLAHE preprocessing

The best-performing variant in the paper. Same network as V2 (MST + FPN +
CBAM + Mask R-CNN), but every input image is first run through Contrast
Limited Adaptive Histogram Equalization (CLAHE) on the L channel of LAB.
That single change pushes detection mAP@0.5 from 0.83 (V2) to 0.89.

## Architecture

```
Input (224 x 224, RGB)
  -> CLAHE on LAB-L (clipLimit=2.0, tileGrid=8x8)   <-- new in V3
  -> MST backbone (Swin-Base, 4 stages)
  -> FPN (out channels = 256)
  -> CBAM (channel + spatial attention)
  -> Mask R-CNN (RPN, ROI Align, box head, mask head)
  -> Detection + Segmentation
```

The CLAHE transform is in [`../common/transforms.py`](../common/transforms.py).
The network reuses [`../tram_v2/model.py`](../tram_v2/model.py) — V3 differs
only at the input pipeline.

## Training

```bash
python -m final.tram_v3.train \
    --data-root /path/to/SSS_OD-5 \
    --epochs 100 \
    --patience 20 \
    --output-dir runs/tram_v3
```

CLAHE knobs: `--clahe-clip` (default 2.0), `--clahe-grid` (default 8).

## Inference

```bash
python -m final.tram_v3.inference \
    --weights runs/tram_v3/tram_v3_best.pth \
    --data-root /path/to/SSS_OD-5/valid \
    --output-dir runs/tram_v3/inference \
    --evaluate-map
```

## Reported numbers (paper)

| Metric                | Value      |
|-----------------------|------------|
| Detection mAP@0.5     | **0.8896** |
| Detection mAP@0.5:0.95| 0.5547     |
| Segmentation mAP@0.5  | **0.7626** |
| Segmentation mAP@0.5:0.95 | 0.4234 |
| Epochs to converge    | 62         |
