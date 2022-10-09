# Copyright (c) Meta Platforms, Inc. and affiliates
from re import L
import torch
from torch.nn import functional as F
from typing import List, Tuple

from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.utils.events import get_event_storage
from detectron2.layers import cat, cross_entropy, nonzero_tuple, batched_nms
from detectron2.structures import Instances, Boxes
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers, _log_classification_stats
)
from cubercnn.modeling.proposal_generator.rpn import matched_pairwise_iou

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4

    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]

    scores_full = scores[filter_inds[:, 0]]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    boxes, scores, filter_inds, scores_full = boxes[keep], scores[keep], filter_inds[keep], scores_full[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.scores_full = scores_full
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(FastRCNNOutputLayers):

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
            
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)


        normalize_factor = max(gt_classes.numel(), 1.0)

        '''
        Standard Faster R-CNN losses
        '''
        _log_classification_stats(scores, gt_classes)
        loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
        loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes, reduction="none")
        loss_box_reg = (loss_box_reg).sum() / normalize_factor

        losses = {
            "BoxHead/loss_cls": loss_cls,
            "BoxHead/loss_box_reg": loss_box_reg,
        }
        
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, reduction='mean'):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if reduction == 'mean':
            if self.box_reg_loss_type == "smooth_l1":
                gt_pred_deltas = self.box2box_transform.get_deltas(
                    proposal_boxes[fg_inds],
                    gt_boxes[fg_inds],
                )
                loss_box_reg = smooth_l1_loss(
                    fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
                )
            elif self.box_reg_loss_type == "giou":
                fg_pred_boxes = self.box2box_transform.apply_deltas(
                    fg_pred_deltas, proposal_boxes[fg_inds]
                )
                loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
            else:
                raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        
            # The reg loss is normalized using the total number of regions (R), not the number
            # of foreground regions even though the box regression loss is only defined on
            # foreground regions. Why? Because doing so gives equal training influence to
            # each foreground example. To see how, consider two different minibatches:
            #  (1) Contains a single foreground region
            #  (2) Contains 100 foreground regions
            # If we normalize by the number of foreground regions, the single example in
            # minibatch (1) will be given 100 times as much influence as each foreground
            # example in minibatch (2). Normalizing by the total number of regions, R,
            # means that the single example in minibatch (1) and each of the 100 examples
            # in minibatch (2) are given equal influence.
            return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty
        
        elif reduction == 'none':
            if self.box_reg_loss_type == "smooth_l1":
                gt_pred_deltas = self.box2box_transform.get_deltas(
                    proposal_boxes[fg_inds],
                    gt_boxes[fg_inds],
                )
                loss_box_reg = smooth_l1_loss(
                    fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="none"
                )
            else:
                raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
            
            # return non-reduced type
            return loss_box_reg
        
        else:
            raise ValueError(f"Invalid bbox reg reduction type '{reduction}'")

    