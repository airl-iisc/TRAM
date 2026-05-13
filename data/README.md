# Dataset

The TRAM models are trained on a SeabedObjects-KLSG-derived side-scan sonar
dataset (`SSS_OD`) with two object classes: `plane` and `ship`. The split used
in the paper is published on Roboflow (workspace `imageenhancement`, project
`sss_od`, version 5).

## Directory layout

The trainers and inference scripts expect a standard YOLOv5-polygon layout:

```
SSS_OD-5/
├── train/
│   ├── images/           # *.jpg or *.png
│   └── labels/           # *.txt (one per image, polygon-format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Pass the parent directory to training as `--data-root`, and the leaf folder
(e.g. `SSS_OD-5/valid`) to inference.

## Label format

Each `<image>.txt` has one line per object:

```
<class_idx> x1 y1 x2 y2 ... xn yn
```

with normalized polygon coordinates in `[0, 1]`. `class_idx` is `0` for plane
and `1` for ship. Internally we shift indices by `+1` so `0` is reserved for
background, matching the Mask R-CNN convention. Bounding boxes are computed
from polygon extents at load time.

## Downloading via Roboflow

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("imageenhancement").project("sss_od")
version = project.version(5)
dataset = version.download("yolov5")    # produces SSS_OD-5/
```

A snapshot is also archived in the parent `datasets/` folder of this repo
(`datasets/SSS_OD-5.tar.gz` and the unpacked `datasets/SSS_OD-5/`).

## Image size and channels

Images are resized to **224 x 224** and converted to 3-channel RGB (Mask R-CNN
expects RGB even for grayscale sonar inputs). Normalization uses ImageNet
statistics, since the Swin backbone is initialized with ImageNet-pretrained
weights.
