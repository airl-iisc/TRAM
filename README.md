# TRAM: Transformer-Based Mask R-CNN Framework for Underwater Object Detection

Reference code for **TRAM: Transformer-Based Mask R-CNN Framework for
Underwater Object Detection in Side-Scan Sonar Data** (Makam, Sundaram, &
Sundaram). This folder is the canonical, paper-aligned drop intended for
reviewers and other researchers who want to reproduce the results or build on
top of the framework.

The rest of the repo (`Experiments/`, `inference/`, `datasets/`, …) contains
the full ablation history. **Use this `final/` folder as the entry point.**

## Three stages, three folders

The paper introduces three progressively enhanced variants. Each lives in its
own folder with a self-contained training and inference entry point:

| Variant | Folder           | Architecture                                  | Det mAP@0.5 | Seg mAP@0.5 |
|---------|------------------|------------------------------------------------|-------------|-------------|
| TRAM V1 | [`tram_v1/`](tram_v1/) | MST + FPN + Mask R-CNN                  | 0.8657      | 0.7226      |
| TRAM V2 | [`tram_v2/`](tram_v2/) | V1 + CBAM (channel + spatial attention) | 0.8321      | 0.7352      |
| TRAM V3 | [`tram_v3/`](tram_v3/) | V2 + CLAHE preprocessing                | **0.8896**  | **0.7626**  |

MST = Modified Swin Transformer (`swin_base_patch4_window7_224` from `timm`,
features-only mode). All variants use 224 x 224 RGB input and the same
Mask R-CNN head configuration.

## Layout

```
final/
├── README.md                  <- you are here
├── requirements.txt
├── common/                    <- shared building blocks
│   ├── dataset.py             <- SSS_OD dataset loader, weighted sampler
│   ├── cbam.py                <- CBAM module (used by V2, V3)
│   ├── transforms.py          <- CLAHETransform (used by V3)
│   ├── coco_eval.py           <- COCO mAP evaluator
│   └── visualization.py       <- box / mask drawing helpers
├── data/
│   └── README.md              <- dataset layout and download instructions
├── tram_v1/                   <- MST + FPN + Mask R-CNN
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── README.md
├── tram_v2/                   <- V1 + CBAM
│   └── ...
└── tram_v3/                   <- V2 + CLAHE
    └── ...
```

## Setup

```bash
# Python 3.10 recommended; 3.9-3.12 should work
python -m venv .venv && source .venv/bin/activate
pip install -r final/requirements.txt
```

## Reproducing the paper

1. Acquire the dataset (see [`data/README.md`](data/README.md)). You should
   end up with a folder containing `train/`, `valid/`, and `test/`
   subdirectories, each with `images/` and `labels/`.

2. Train each variant from the repo root:

   ```bash
   python -m final.tram_v1.train --data-root /path/to/SSS_OD-5 --output-dir runs/tram_v1
   python -m final.tram_v2.train --data-root /path/to/SSS_OD-5 --output-dir runs/tram_v2
   python -m final.tram_v3.train --data-root /path/to/SSS_OD-5 --output-dir runs/tram_v3
   ```

   Each writes a `*_best.pth` (highest validation bbox mAP@0.5:0.95) and a
   per-epoch `training.log`.

3. Run inference and compute mAP:

   ```bash
   python -m final.tram_v1.inference \
       --weights runs/tram_v1/tram_v1_best.pth \
       --data-root /path/to/SSS_OD-5/valid \
       --evaluate-map
   ```

   Same pattern for V2 and V3.

## Hardware

The paper trained on a single NVIDIA RTX 4090 (24 GB) and Kaggle's NVIDIA
Tesla P100 (16 GB). Batch size 4 fits comfortably in 16 GB at 224 x 224.

## Trained weights

Trained checkpoints can be accessed here: https://drive.google.com/drive/folders/1tRt5SZSNibpIC6tU9roY_KQUPiBM3rvb?usp=drive_link 

## Citation

If you use this code, please cite:

```
@inproceedings{Makam2025TRAM,
  author={Makam, Rajini and Sundaram, Kalyana and Sundaram, Suresh},
  booktitle={OCEANS 2025 Brest}, 
  title={TRAM: Transformer-Based Mask R-CNN Framework for Underwater Object Detection in Side-Scan Sonar Data}, 
  year={2025},
  pages={1-6},
  doi={10.1109/OCEANS58557.2025.11104722}
}
```
