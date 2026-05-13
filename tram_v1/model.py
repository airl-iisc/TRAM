"""TRAM V1: Modified Swin Transformer (MST) + FPN + Mask R-CNN.

Backbone: timm `swin_base_patch4_window7_224` in features-only mode.
The four feature maps (56x56x128, 28x28x256, 14x14x512, 7x7x1024) are passed
through a Feature Pyramid Network and then into Mask R-CNN's RPN, ROI Align,
box head, and mask head.
"""

from collections import OrderedDict

import timm
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign


class MSTBackbone(nn.Module):
    """Modified Swin Transformer backbone with FPN."""

    def __init__(self, pretrained: bool = True, fpn_out_channels: int = 256):
        super().__init__()
        self.body = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=224,
        )
        in_channels_list = self.body.feature_info.channels()
        self.out_channels = fpn_out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=None,
        )

    def forward(self, x):
        feats = self.body(x)
        features = OrderedDict()
        for idx, feature in enumerate(feats):
            # Swin returns [B, H, W, C]; FPN expects [B, C, H, W].
            if feature.dim() == 4:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            features[str(idx)] = feature
        return self.fpn(features)


def build_model(num_classes: int = 3, pretrained: bool = True) -> MaskRCNN:
    """Build TRAM V1 (MST + FPN + Mask R-CNN).

    Args:
        num_classes: includes background. Default 3 = background + plane + ship.
        pretrained: load ImageNet-pretrained Swin weights.
    """
    backbone = MSTBackbone(pretrained=pretrained)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    return MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        ),
        mask_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
        ),
        min_size=224,
        max_size=224,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
