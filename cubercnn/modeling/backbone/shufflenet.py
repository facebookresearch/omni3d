# Copyright (c) Meta Platforms, Inc. and affiliates
from torchvision import models
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN

class ShufflenetBackbone(Backbone):
    def __init__(self, cfg, input_shape, pretrained=True):
        super().__init__()

        base  = models.shufflenet_v2_x1_0(pretrained)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5

        self._out_feature_channels = {'p2': 24, 'p3': 116, 'p4': 232, 'p5': 464, 'p6': 464}
        self._out_feature_strides ={'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self._out_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    
    def forward(self, x):

        outputs = {}
        
        x = self.conv1(x)
        p2 = self.maxpool(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)

        outputs['p2'] = p2
        outputs['p3'] = p3
        outputs['p4'] = p4
        outputs['p5'] = p5
        outputs['p6'] = p6

        return outputs


@BACKBONE_REGISTRY.register()
def build_shufflenet_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''
    
    bottom_up = ShufflenetBackbone(cfg, input_shape, pretrained=imagenet_pretrain)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
