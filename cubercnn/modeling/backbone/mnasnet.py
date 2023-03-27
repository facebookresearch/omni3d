# Copyright (c) Meta Platforms, Inc. and affiliates
from torchvision import models
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN

class MNASNetBackbone(Backbone):
    def __init__(self, cfg, input_shape, pretrained=True):
        super().__init__()

        base  = models.mnasnet1_0(pretrained)
        base  = base.layers 
        
        self.base = base

        self._out_feature_channels = {'p2': 24, 'p3': 40, 'p4': 96, 'p5': 320, 'p6': 320}
        self._out_feature_strides ={'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self._out_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    
    def forward(self, x):

        outputs = {}
        
        p2 = self.base[0:9](x)
        p3 = self.base[9](p2)
        p4 = self.base[10:12](p3)
        p5 = self.base[12:14](p4)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)
        outputs['p2'] = p2
        outputs['p3'] = p3
        outputs['p4'] = p4
        outputs['p5'] = p5
        outputs['p6'] = p6

        return outputs

@BACKBONE_REGISTRY.register()
def build_mnasnet_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''

    bottom_up = MNASNetBackbone(cfg, input_shape, pretrained=imagenet_pretrain)
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
