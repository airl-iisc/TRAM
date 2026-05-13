# TRAM V2 — V1 + CBAM

Builds on V1 by inserting a Convolutional Block Attention Module (CBAM) after
the FPN. CBAM applies channel attention then spatial attention, helping the
network focus on relevant objects against cluttered seabed backgrounds.

## Architecture

```
Input (224 x 224, RGB)
  -> MST backbone (Swin-Base, 4 stages)
  -> FPN (out channels = 256)
  -> CBAM (channel + spatial attention)   <-- new in V2
  -> Mask R-CNN (RPN, ROI Align, box head, mask head)
  -> Detection + Segmentation
```

CBAM lives in [`../common/cbam.py`](../common/cbam.py). The model wiring is in
[`model.py`](model.py).

## Training

```bash
python -m final.tram_v2.train \
    --data-root /path/to/SSS_OD-5 \
    --epochs 100 \
    --patience 20 \
    --output-dir runs/tram_v2
```

Differences from V1 training:
- Cosine annealing LR schedule (`T_max = epochs`)
- Mixed precision (AMP) on CUDA
- Early stopping after `--patience` epochs without bbox mAP improvement

## Inference

```bash
python -m final.tram_v2.inference \
    --weights runs/tram_v2/tram_v2_best.pth \
    --data-root /path/to/SSS_OD-5/valid \
    --output-dir runs/tram_v2/inference \
    --evaluate-map
```

## Reported numbers (paper)

| Metric                | Value  |
|-----------------------|--------|
| Detection mAP@0.5     | 0.8321 |
| Detection mAP@0.5:0.95| 0.5293 |
| Segmentation mAP@0.5  | 0.7352 |
| Segmentation mAP@0.5:0.95 | 0.4412 |
| Epochs to converge    | 44     |
