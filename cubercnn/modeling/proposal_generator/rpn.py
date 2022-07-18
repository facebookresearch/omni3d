# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List, Tuple
import torch
from typing import List, Tuple, Union
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, Instances, pairwise_iou, pairwise_ioa
from detectron2.utils.memory import retry_if_cuda_oom
from fvcore.nn import smooth_l1_loss
from detectron2.layers import cat
from detectron2.layers import nonzero_tuple

from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY

@PROPOSAL_GENERATOR_REGISTRY.register()
class RPNWithIgnore(RPN):
    
    @configurable
    def __init__(
        self,
        *,
        ignore_thresh: float = 0.5,
        objectness_uncertainty: str = 'none',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ignore_thresh = ignore_thresh
        self.objectness_uncertainty = objectness_uncertainty

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["ignore_thresh"] = cfg.MODEL.RPN.IGNORE_THRESHOLD
        ret["objectness_uncertainty"] = cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY 
        return ret
    
    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        anchors = Boxes.cat(anchors)

        # separate valid and ignore gts
        gt_boxes_ign = [x.gt_boxes[x.gt_classes < 0] for x in gt_instances]
        gt_boxes = [x.gt_boxes[x.gt_classes >= 0] for x in gt_instances]

        del gt_instances

        gt_labels = []
        matched_gt_boxes = []

        for gt_boxes_i, gt_boxes_ign_i in zip(gt_boxes, gt_boxes_ign):
            """
            gt_boxes_i: ground-truth boxes for i-th image
            gt_boxes_ign_i: ground-truth ignore boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)

            gt_arange = torch.arange(match_quality_matrix.shape[1]).to(matched_idxs.device)
            matched_ious = match_quality_matrix[matched_idxs, gt_arange]
            
            best_ious_gt_vals, best_ious_gt_ind = match_quality_matrix.max(dim=1)

            del match_quality_matrix

            best_inds = torch.tensor(list(set(best_ious_gt_ind.tolist()) & set((gt_labels_i == 1).nonzero().squeeze(1).tolist())))

            # A vector of labels (-1, 0, 1) for each anchor
            # which denote (ignore, background, foreground)
            gt_labels_i = self._subsample_labels(gt_labels_i, matched_ious=matched_ious)

            # overrride the best possible GT options, always selected for sampling.
            # otherwise aggressive thresholds may produce HUGE amounts of low quality FG.
            if best_inds.numel() > 0:
                gt_labels_i[best_inds] = 1.0

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            if len(gt_boxes_ign_i) > 0: 

                # compute the quality matrix, only on subset of background
                background_inds = (gt_labels_i == 0).nonzero().squeeze()

                if background_inds.numel() > 1:
                    
                    match_quality_matrix_ign = retry_if_cuda_oom(pairwise_ioa)(gt_boxes_ign_i, anchors[background_inds])

                    # determine the boxes inside ignore regions with sufficient threshold
                    gt_labels_i[background_inds[match_quality_matrix_ign.max(0)[0] >= self.ignore_thresh]] = -1
                
                    del match_quality_matrix_ign

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes
    
    def _subsample_labels(self, label, matched_ious=None):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.
        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0, matched_ious=matched_ious
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        if not self.objectness_uncertainty.lower() in ['none']:
            localization_loss, objectness_loss = _dense_box_regression_loss_with_uncertainty(
                anchors,
                self.box2box_transform,
                pred_anchor_deltas,
                pred_objectness_logits,
                gt_boxes,
                pos_mask,
                box_reg_loss_type=self.box_reg_loss_type,
                smooth_l1_beta=self.smooth_l1_beta,
                uncertainty_type=self.objectness_uncertainty,
            )
        else:
            localization_loss = _dense_box_regression_loss(
                anchors,
                self.box2box_transform,
                pred_anchor_deltas,
                gt_boxes,
                pos_mask,
                box_reg_loss_type=self.box_reg_loss_type,
                smooth_l1_beta=self.smooth_l1_beta,
            )

            valid_mask = gt_labels >= 0
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask],
                gt_labels[valid_mask].to(torch.float32),
                reduction="sum",
            )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "rpn/cls": objectness_loss / normalizer,
            "rpn/loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

def _dense_box_regression_loss_with_uncertainty(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
    uncertainty_type='centerness',
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.
    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)

    n = len(gt_boxes)
    
    boxes_fg = Boxes(anchors.unsqueeze(0).repeat([n, 1, 1])[fg_mask])
    gt_boxes_fg = Boxes(torch.stack(gt_boxes)[fg_mask].detach())
    objectness_targets_anchors = matched_pairwise_iou(boxes_fg, gt_boxes_fg).detach()

    objectness_logits = torch.cat(pred_objectness_logits, dim=1)

    # Numerically the same as (-(y*torch.log(p) + (1 - y)*torch.log(1 - p))).sum()
    loss_box_conf = F.binary_cross_entropy_with_logits(
        objectness_logits[fg_mask], 
        objectness_targets_anchors,
        reduction='none'
    )

    loss_box_conf = (loss_box_conf * objectness_targets_anchors).sum()
    
    # keep track of how scores look for FG / BG.
    # ideally, FG slowly >>> BG scores as regression improves. 
    storage = get_event_storage()
    storage.put_scalar("rpn/conf_pos_anchors", torch.sigmoid(objectness_logits[fg_mask]).mean().item())
    storage.put_scalar("rpn/conf_neg_anchors", torch.sigmoid(objectness_logits[~fg_mask]).mean().item())

    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction="none",
        )
        
        loss_box_reg = (loss_box_reg.sum(dim=1) * objectness_targets_anchors).sum()

    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")

    return loss_box_reg, loss_box_conf

def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int, matched_ious=None, eps=1e-4
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.
    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.
    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    #if positive_fraction == 1.0 and num_neg > 10:
    # allow some negatives for statistics only.
    #num_neg = 10
    
    # randomly select positive and negative examples
    if num_pos > 0 and matched_ious is not None:
        perm1 = torch.multinomial(matched_ious[positive] + eps, num_pos)
    else:
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    if num_neg > 0 and matched_ious is not None:
        perm2 = torch.multinomial(matched_ious[negative] + eps, num_neg)
    else:
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes that have the same number of boxes.
    Similar to :func:`pairwise_iou`, but computes only diagonal elements of the matrix.
    Args:
        boxes1 (Boxes): bounding boxes, sized [N,4].
        boxes2 (Boxes): same length as boxes1
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou