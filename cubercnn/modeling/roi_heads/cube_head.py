# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.utils.registry import Registry
from typing import Dict
from detectron2.layers import ShapeSpec
from torch import nn
import torch
import numpy as np
import fvcore.nn.weight_init as weight_init

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)

ROI_CUBE_HEAD_REGISTRY = Registry("ROI_CUBE_HEAD")

@ROI_CUBE_HEAD_REGISTRY.register()
class CubeHead(nn.Module):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        #-------------------------------------------
        # Settings
        #-------------------------------------------
        self.num_classes        = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.use_conf           = cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE
        self.z_type             = cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE
        self.pose_type          = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        self.cluster_bins       = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
        self.shared_fc          = cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC

        #-------------------------------------------
        # Feature generator
        #-------------------------------------------

        num_conv = cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_CUBE_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_CUBE_HEAD.FC_DIM

        conv_dims = [conv_dim] * num_conv
        fc_dims = [fc_dim] * num_fc

        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        if self.shared_fc:
            self.feature_generator = nn.Sequential()
        else:
            self.feature_generator_XY = nn.Sequential()
            self.feature_generator_dims = nn.Sequential()
            self.feature_generator_pose = nn.Sequential()
            self.feature_generator_Z = nn.Sequential()

            if self.use_conf:
                self.feature_generator_conf = nn.Sequential()

        # create fully connected layers for Cube Head
        for k, fc_dim in enumerate(fc_dims):
            
            fc_dim_in = int(np.prod(self._output_size))
            
            self._output_size = fc_dim

            if self.shared_fc:
                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator.add_module("fc{}".format(k + 1), fc)
                self.feature_generator.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            
            else:
                
                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_dims.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_dims.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_XY.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_XY.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_pose.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_pose.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_Z.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_Z.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                if self.use_conf:
                    fc = nn.Linear(fc_dim_in, fc_dim)
                    weight_init.c2_xavier_fill(fc)
                    self.feature_generator_conf.add_module("fc{}".format(k + 1), fc)
                    self.feature_generator_conf.add_module("fc_relu{}".format(k + 1), nn.ReLU())

        #-------------------------------------------
        # 3D outputs
        #-------------------------------------------
        
        # Dimensions in meters (width, height, length)
        self.bbox_3D_dims = nn.Linear(self._output_size, self.num_classes*3)
        nn.init.normal_(self.bbox_3D_dims.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_dims.bias, 0)

        cluster_bins = self.cluster_bins if self.cluster_bins > 1 else 1

        # XY
        self.bbox_3D_center_deltas = nn.Linear(self._output_size, self.num_classes*2)
        nn.init.normal_(self.bbox_3D_center_deltas.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_deltas.bias, 0)

        # Pose
        if self.pose_type == '6d':
            self.bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*6)

        elif self.pose_type == 'quaternion':
            self.bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*4)

        elif self.pose_type == 'euler':
            self.bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*3)

        else:
            raise ValueError('Cuboid pose type {} is not recognized'.format(self.pose_type))
        
        nn.init.normal_(self.bbox_3D_pose.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_pose.bias, 0)

        # Z 
        self.bbox_3D_center_depth = nn.Linear(self._output_size, self.num_classes*cluster_bins)
        nn.init.normal_(self.bbox_3D_center_depth.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_depth.bias, 0)

        # Optionally, box confidence
        if self.use_conf:
            self.bbox_3D_uncertainty = nn.Linear(self._output_size, self.num_classes*1)
            nn.init.normal_(self.bbox_3D_uncertainty.weight, std=0.001)
            nn.init.constant_(self.bbox_3D_uncertainty.bias, 5)


    def forward(self, x):
    
        n = x.shape[0]
        
        box_z = None
        box_uncert = None
        box_2d_deltas = None

        if self.shared_fc:
            features = self.feature_generator(x)
            box_2d_deltas = self.bbox_3D_center_deltas(features)
            box_dims = self.bbox_3D_dims(features)
            box_pose = self.bbox_3D_pose(features)
            box_z = self.bbox_3D_center_depth(features)

            if self.use_conf:
                box_uncert = self.bbox_3D_uncertainty(features).clip(0.01)
        else:

            box_2d_deltas = self.bbox_3D_center_deltas(self.feature_generator_XY(x))
            box_dims = self.bbox_3D_dims(self.feature_generator_dims(x))
            box_pose = self.bbox_3D_pose(self.feature_generator_pose(x))
            box_z = self.bbox_3D_center_depth(self.feature_generator_Z(x))

            if self.use_conf:
                box_uncert = self.bbox_3D_uncertainty(self.feature_generator_conf(x)).clip(0.01)

        # Pose
        if self.pose_type == '6d':
            box_pose = rotation_6d_to_matrix(box_pose.view(-1, 6))

        elif self.pose_type == 'quaternion':
            quats = box_pose.view(-1, 4)
            quats_scales = (quats * quats).sum(1)
            quats = quats / _copysign(torch.sqrt(quats_scales), quats[:, 0])[:, None]
            box_pose = quaternion_to_matrix(quats)

        elif self.pose_type == 'euler':
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'XYZ')

        box_2d_deltas = box_2d_deltas.view(n, self.num_classes, 2)
        box_dims = box_dims.view(n, self.num_classes, 3)
        box_pose = box_pose.view(n, self.num_classes, 3, 3)

        if self.cluster_bins > 1:
            box_z = box_z.view(n, self.cluster_bins, self.num_classes, -1)

        else:
            box_z = box_z.view(n, self.num_classes, -1)
            
        return box_2d_deltas, box_z, box_dims, box_pose, box_uncert


def build_cube_head(cfg, input_shape: Dict[str, ShapeSpec]):
    name = cfg.MODEL.ROI_CUBE_HEAD.NAME
    return ROI_CUBE_HEAD_REGISTRY.get(name)(cfg, input_shape)