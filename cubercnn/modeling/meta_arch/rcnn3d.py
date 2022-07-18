# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List, Optional
import torch
import numpy as np
from detectron2.layers import ShapeSpec

from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import _log_api_usage
from detectron2.modeling.meta_arch import (
    META_ARCH_REGISTRY, GeneralizedRCNN
)
from cubercnn.modeling.roi_heads import build_roi_heads

@META_ARCH_REGISTRY.register()
class RCNN3D(GeneralizedRCNN):
    
    @classmethod
    def from_config(cls, cfg, priors=None):
        backbone = build_backbone(cfg, priors=priors)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape(), priors=priors),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [info['height'] / im.shape[1] for (info, im) in zip(batched_inputs, images)]

        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        _, detector_losses = self.roi_heads(
            images, features, proposals, 
            Ks, im_scales_ratio, 
            gt_instances
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [info['height'] / im.shape[1] for (info, im) in zip(batched_inputs, images)]
        
        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]

        features = self.backbone(images.tensor)

        # Pass oracle 2D boxes into the RoI heads
        if type(batched_inputs == list) and np.any(['oracle2D' in b for b in batched_inputs]):
            oracles = [b['oracle2D'] for b in batched_inputs]
            results, _ = self.roi_heads(images, features, oracles, Ks, im_scales_ratio, None)
        
        # normal inference
        else:
            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, Ks, im_scales_ratio, None)
            
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

def build_model(cfg, priors=None):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model


def build_backbone(cfg, input_shape=None, priors=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape, priors)
    assert isinstance(backbone, Backbone)
    return backbone