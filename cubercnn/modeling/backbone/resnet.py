# Copyright (c) Meta Platforms, Inc. and affiliates
from torchvision import models
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN

class ResNet(Backbone):
    def __init__(self, cfg, input_shape, pretrained=True):
        super().__init__()

        if cfg.MODEL.RESNETS.DEPTH == 18:
            base  = models.resnet18(pretrained)
            self._out_feature_channels = {'p2': 64, 'p3': 128, 'p4': 256, 'p5': 512, 'p6': 512}
        elif cfg.MODEL.RESNETS.DEPTH == 34:
            base  = models.resnet34(pretrained)
            self._out_feature_channels = {'p2': 64, 'p3': 128, 'p4': 256, 'p5': 512, 'p6': 512}
        elif cfg.MODEL.RESNETS.DEPTH == 50:
            base  = models.resnet50(pretrained)
            self._out_feature_channels = {'p2': 256, 'p3': 512, 'p4': 1024, 'p5': 2048, 'p6': 2048}
        elif cfg.MODEL.RESNETS.DEPTH == 101:
            base  = models.resnet101(pretrained)
            self._out_feature_channels = {'p2': 256, 'p3': 512, 'p4': 1024, 'p5': 2048, 'p6': 2048}
        else:
            raise ValueError('No configuration currently supporting depth of {}'.format(cfg.MODEL.RESNETS.DEPTH))
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self._out_feature_strides ={'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self._out_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    
    def forward(self, x):

        outputs = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        p2 = self.layer1(x)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)

        outputs['p2'] = p2
        outputs['p3'] = p3
        outputs['p4'] = p4
        outputs['p5'] = p5
        outputs['p6'] = p6

        return outputs


@BACKBONE_REGISTRY.register()
def build_resnet_from_vision_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''

    if cfg.MODEL.RESNETS.TORCHVISION:
        bottom_up = ResNet(cfg, input_shape, pretrained=imagenet_pretrain)

    else:
        # use the MSRA modeling logic to build the backbone.
        bottom_up = build_resnet_backbone(cfg, input_shape)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
