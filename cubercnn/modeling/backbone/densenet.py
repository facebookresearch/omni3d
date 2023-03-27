# Copyright (c) Meta Platforms, Inc. and affiliates
from torchvision import models
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN

class DenseNetBackbone(Backbone):
    def __init__(self, cfg, input_shape, pretrained=True):
        super().__init__()

        base  = models.densenet121(pretrained)
        base  = base.features

        self.base = base
        
        self._out_feature_channels = {'p2': 256, 'p3': 512, 'p4': 1024, 'p5': 1024, 'p6': 1024}
        self._out_feature_strides ={'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self._out_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    
    def forward(self, x):

        outputs = {}
        
        db1 = self.base[0:5](x)
        db2 = self.base[5:7](db1)
        db3 = self.base[7:9](db2)
        p5 = self.base[9:](db3)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)
        outputs['p2'] = db1
        outputs['p3'] = db2
        outputs['p4'] = db3
        outputs['p5'] = p5
        outputs['p6'] = p6

        return outputs


@BACKBONE_REGISTRY.register()
def build_densenet_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''

    bottom_up = DenseNetBackbone(cfg, input_shape, pretrained=imagenet_pretrain)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE
    )
    return backbone