# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import detectron2.utils.comm as comm

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN

BatchNorm = nn.BatchNorm2d

"""
Adapted models from repositories
    Deep Layer Aggregation CVPR 2018
    https://github.com/ucbdrive/dla
    BSD-3 Licence https://github.com/ucbdrive/dla/blob/master/LICENSE
    
    Geometry Uncertainty Projection Network for Monocular 3D Object Detection, ICCV 2021
    https://github.com/SuperMHP/GUPNet/blob/main/code/lib/backbones/dla.py
    MIT Licence https://github.com/SuperMHP/GUPNet/blob/main/LICENSE
"""

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return os.path.join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)


    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        
        # load model only on main process
        # to prevent redundent model caching
        if comm.is_main_process():
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
            del model_weights['fc.weight']
            del model_weights['fc.bias']
            self.load_state_dict(model_weights)


def dla34(pretrained=False, tricks=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        if tricks:
            model.load_pretrained_model(data='imagenet', name='dla34+tricks', hash='24a49e58')
        else:
            model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def dla46_c(pretrained=False, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla46_c', hash='2bfd52c3')
    return model


def dla46x_c(pretrained=False, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla46x_c', hash='d761bae7')
    return model


def dla60x_c(pretrained=False, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=False, tricks=False, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained:
        if tricks:
            model.load_pretrained_model(data='imagenet', name='dla60+tricks', hash='14488826')
        else:
            model.load_pretrained_model(data='imagenet', name='dla60', hash='24839fc4')

    return model


def dla60x(pretrained=False, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x', hash='d15cacda')
    return model


def dla102(pretrained=False, tricks=False, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:

        if tricks:
            model.load_pretrained_model(data='imagenet', name='dla102+tricks', hash='27a30eac')
        else:
            model.load_pretrained_model(data='imagenet', name='dla102', hash='d94d9790')
    return model


def dla102x(pretrained=False, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102x', hash='ad62be81')
    return model


def dla102x2(pretrained=False, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102x2', hash='262837b6')
    return model


def dla169(pretrained=False, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla169', hash='0914e092')
    return model

class DLABackbone(Backbone):
    def __init__(self, cfg, input_shape, pretrained=True):
        super().__init__()

        if cfg.MODEL.DLA.TYPE == 'dla34':
            base  = dla34(pretrained=pretrained, tricks=cfg.MODEL.DLA.TRICKS)
            self._out_feature_channels = {'p2': 64, 'p3': 128, 'p4': 256, 'p5': 512, 'p6': 512}
        elif cfg.MODEL.DLA.TYPE == 'dla46_c':
            base  = dla46_c(pretrained=pretrained)
            self._out_feature_channels = {'p2': 64, 'p3': 64, 'p4': 128, 'p5': 256, 'p6': 256}
        elif cfg.MODEL.DLA.TYPE == 'dla46x_c':
            base  = dla46x_c(pretrained=pretrained)
            self._out_feature_channels = {'p2': 64, 'p3': 64, 'p4': 128, 'p5': 256, 'p6': 256}
        elif cfg.MODEL.DLA.TYPE == 'dla60x_c':
            base  = dla60x_c(pretrained=pretrained)
            self._out_feature_channels = {'p2': 64, 'p3': 64, 'p4': 128, 'p5': 256, 'p6': 256}
        elif cfg.MODEL.DLA.TYPE == 'dla60':
            base  = dla60(pretrained=pretrained, tricks=cfg.MODEL.DLA.TRICKS)
            self._out_feature_channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 1024, 'p6': 1024}
        elif cfg.MODEL.DLA.TYPE == 'dla60x':
            base  = dla60x(pretrained=pretrained)
            self._out_feature_channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 1024, 'p6': 1024}
        elif cfg.MODEL.DLA.TYPE == 'dla102':
            base  = dla102(pretrained=pretrained, tricks=cfg.MODEL.DLA.TRICKS)
            self._out_feature_channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 1024, 'p6': 1024}
        elif cfg.MODEL.DLA.TYPE == 'dla102x':
            base  = dla102x(pretrained=pretrained)
            self._out_feature_channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 1024, 'p6': 1024}
        elif cfg.MODEL.DLA.TYPE == 'dla102x2':
            base  = dla102x2(pretrained=pretrained)
            self._out_feature_channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 1024, 'p6': 1024}
        elif cfg.MODEL.DLA.TYPE == 'dla169':
            base  = dla169(pretrained=pretrained)
            self._out_feature_channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 1024, 'p6': 1024}

        self.base_layer = base.base_layer
        self.level0 = base.level0
        self.level1 = base.level1
        self.level2 = base.level2
        self.level3 = base.level3
        self.level4 = base.level4
        self.level5 = base.level5
        
        self._out_feature_strides ={'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self._out_features = ['p2', 'p3', 'p4', 'p5', 'p6']
    
    def forward(self, x):

        outputs = {}
        
        base_layer = self.base_layer(x)
        level0 = self.level0(base_layer)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        level4 = self.level4(level3)
        level5 = self.level5(level4)
        level6 = F.max_pool2d(level5, kernel_size=1, stride=2, padding=0)

        outputs['p2'] = level2
        outputs['p3'] = level3
        outputs['p4'] = level4
        outputs['p5'] = level5
        outputs['p6'] = level6

        return outputs

@BACKBONE_REGISTRY.register()
def build_dla_from_vision_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''

    bottom_up = DLABackbone(cfg, input_shape, pretrained=imagenet_pretrain)
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