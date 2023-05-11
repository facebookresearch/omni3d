# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import copy
import datetime
import io
import itertools
import json
import logging
import os
import time
from collections import defaultdict
from typing import List, Union
from typing import Tuple

import numpy as np
import pycocotools.mask as maskUtils
import torch
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from detectron2.utils.comm import get_world_size, is_main_process
import detectron2.utils.comm as comm
from detectron2.evaluation import (
    DatasetEvaluators, inference_context, DatasetEvaluator
)
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from torch import nn

import logging
from cubercnn.data import Omni3D
from pytorch3d import _C
import torch.nn.functional as F

from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles

import cubercnn.vis.logperf as utils_logperf
from cubercnn.data import (
    get_omni3d_categories,
    simple_register
)

"""
This file contains
* Omni3DEvaluationHelper: a helper object to accumulate and summarize evaluation results
* Omni3DEval: a wrapper around COCOeval to perform 3D bounding evaluation in the detection setting
* Omni3DEvaluator: a wrapper around COCOEvaluator to collect results on each dataset
* Omni3DParams: parameters for the evaluation API
"""

logger = logging.getLogger(__name__)

# Defines the max cross of len(dts) * len(gts)
# which we will attempt to compute on a GPU. 
# Fallback is safer computation on a CPU. 
# 0 is disabled on GPU. 
MAX_DTS_CROSS_GTS_FOR_IOU3D = 0


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
    """
    Checks that plane vertices are coplanar.
    Returns a bool tensor of size B, where True indicates a box is coplanar.
    """
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    
    return (mat1.bmm(mat2).abs() < eps).view(B)


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> torch.BoolTensor:
    """
    Checks that the sides of the box have a non zero area.
    Returns a bool tensor of size B, where True indicates a box is nonzero.
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return (face_areas > eps).all(1).view(B)

def box3d_overlap(
    boxes_dt: torch.Tensor, boxes_gt: torch.Tensor, 
    eps_coplanar: float = 1e-4, eps_nonzero: float = 1e-8
) -> torch.Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero 
    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero  = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print('Warning: skipping {:d} non-coplanar boxes at eval.'.format(int(invalid_coplanar.float().sum())))
    
    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print('Warning: skipping {:d} zero volume boxes at eval.'.format(int(invalid_nonzero.float().sum())))

    return ious

class Omni3DEvaluationHelper:
    def __init__(self, 
            dataset_names, 
            filter_settings, 
            output_folder,
            iter_label='-',
            only_2d=False,
        ):
        """
        A helper class to initialize, evaluate and summarize Omni3D metrics. 

        The evaluator relies on the detectron2 MetadataCatalog for keeping track 
        of category names and contiguous IDs. Hence, it is important to set 
        these variables appropriately. 
        
        # (list[str]) the category names in their contiguous order
        MetadataCatalog.get('omni3d_model').thing_classes = ... 

        # (dict[int: int]) the mapping from Omni3D category IDs to the contiguous order
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

        Args:
            dataset_names (list[str]): the individual dataset splits for evaluation
            filter_settings (dict): the filter settings used for evaluation, see
                cubercnn/data/datasets.py get_filter_settings_from_cfg
            output_folder (str): the output folder where results can be stored to disk.
            iter_label (str): an optional iteration/label used within the summary
            only_2d (bool): whether the evaluation mode should be 2D or 2D and 3D.
        """
        
        self.dataset_names = dataset_names
        self.filter_settings = filter_settings
        self.output_folder = output_folder
        self.iter_label = iter_label
        self.only_2d = only_2d

        # Each dataset evaluator is stored here
        self.evaluators = OrderedDict()

        # These are the main evaluation results
        self.results = OrderedDict()

        # These store store per-dataset results to be printed
        self.results_analysis = OrderedDict()
        self.results_omni3d = OrderedDict()

        self.overall_imgIds = set()
        self.overall_catIds = set()
        
        # These store the evaluations for each category and area,
        # concatenated from ALL evaluated datasets. Doing so avoids
        # the need to re-compute them when accumulating results.
        self.evals_per_cat_area2D = {}
        self.evals_per_cat_area3D = {}
        
        self.output_folders = {
            dataset_name: os.path.join(self.output_folder, dataset_name)
            for dataset_name in dataset_names
        }

        for dataset_name in self.dataset_names:
            
            # register any datasets that need it
            if MetadataCatalog.get(dataset_name).get('json_file') is None:
                simple_register(dataset_name, filter_settings, filter_empty=False)
            
            # create an individual dataset evaluator
            self.evaluators[dataset_name] = Omni3DEvaluator(
                dataset_name, output_dir=self.output_folders[dataset_name], 
                filter_settings=self.filter_settings, only_2d=self.only_2d, 
                eval_prox=('Objectron' in dataset_name or 'SUNRGBD' in dataset_name),
                distributed=False, # actual evaluation should be single process
            )

            self.evaluators[dataset_name].reset()
            self.overall_imgIds.update(set(self.evaluators[dataset_name]._omni_api.getImgIds()))
            self.overall_catIds.update(set(self.evaluators[dataset_name]._omni_api.getCatIds()))
        
    def add_predictions(self, dataset_name, predictions):
        """
        Adds predictions to the evaluator for dataset_name. This can be any number of
        predictions, including all predictions passed in at once or in batches. 

        Args:
            dataset_name (str): the dataset split name which the predictions belong to
            predictions (list[dict]): each item in the list is a dict as follows:

                {
                    "image_id": <int> the unique image identifier from Omni3D,
                    "K": <np.array> 3x3 intrinsics matrix for the image,
                    "width": <int> image width,
                    "height": <int> image height,
                    "instances": [
                        {
                            "image_id":  <int> the unique image identifier from Omni3D,
                            "category_id": <int> the contiguous category prediction IDs, 
                                which can be mapped from Omni3D's category ID's using
                                MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
                            "bbox": [float] 2D box as [x1, y1, x2, y2] used for IoU2D,
                            "score": <float> the confidence score for the object,
                            "depth": <float> the depth of the center of the object,
                            "bbox3D": list[list[float]] 8x3 corner vertices used for IoU3D,
                        }
                        ...
                    ]
                }
        """
        # concatenate incoming predictions
        self.evaluators[dataset_name]._predictions += predictions

    def save_predictions(self, dataset_name):
        """
        Saves the predictions from dataset_name to disk, in a self.output_folder.

        Args:
            dataset_name (str): the dataset split name which should be saved.
        """
        # save predictions to disk
        output_folder_dataset = self.output_folders[dataset_name]
        PathManager.mkdirs(output_folder_dataset)
        file_path = os.path.join(output_folder_dataset, "instances_predictions.pth")
        with PathManager.open(file_path, "wb") as f:
            torch.save(self.evaluators[dataset_name]._predictions, f)

    def evaluate(self, dataset_name):
        """
        Runs the evaluation for an individual dataset split, assuming all 
        predictions have been passed in. 

        Args:
            dataset_name (str): the dataset split name which should be evalated.
        """
        
        if not dataset_name in self.results:
            
            # run evaluation and cache
            self.results[dataset_name] = self.evaluators[dataset_name].evaluate()

        results = self.results[dataset_name]

        logger.info('\n'+results['log_str_2D'].replace('mode=2D', '{} iter={} mode=2D'.format(dataset_name, self.iter_label)))
            
        # store the partially accumulated evaluations per category per area
        for key, item in results['bbox_2D_evals_per_cat_area'].items():
            if not key in self.evals_per_cat_area2D:
                self.evals_per_cat_area2D[key] = []
            self.evals_per_cat_area2D[key] += item

        if not self.only_2d:
            # store the partially accumulated evaluations per category per area
            for key, item in results['bbox_3D_evals_per_cat_area'].items():
                if not key in self.evals_per_cat_area3D:
                    self.evals_per_cat_area3D[key] = []
                self.evals_per_cat_area3D[key] += item

            logger.info('\n'+results['log_str_3D'].replace('mode=3D', '{} iter={} mode=3D'.format(dataset_name, self.iter_label)))

        # full model category names
        category_names = self.filter_settings['category_names']

        # The set of categories present in the dataset; there should be no duplicates 
        categories = {cat for cat in category_names if 'AP-{}'.format(cat) in results['bbox_2D']}
        assert len(categories) == len(set(categories)) 

        # default are all NaN
        general_2D, general_3D, omni_2D, omni_3D = (np.nan,) * 4

        # 2D and 3D performance for categories in dataset; and log
        general_2D = np.mean([results['bbox_2D']['AP-{}'.format(cat)] for cat in categories])
        if not self.only_2d:
            general_3D = np.mean([results['bbox_3D']['AP-{}'.format(cat)] for cat in categories])

        # 2D and 3D performance on Omni3D categories
        omni3d_dataset_categories = get_omni3d_categories(dataset_name)  # dataset-specific categories
        if len(omni3d_dataset_categories - categories) == 0:  # omni3d_dataset_categories is a subset of categories
            omni_2D = np.mean([results['bbox_2D']['AP-{}'.format(cat)] for cat in omni3d_dataset_categories])
            if not self.only_2d:
                omni_3D = np.mean([results['bbox_3D']['AP-{}'.format(cat)] for cat in omni3d_dataset_categories])
        
        self.results_omni3d[dataset_name] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D}

        # Performance analysis
        extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf = (np.nan,)*6
        if not self.only_2d:
            extras_AP15 = results['bbox_3D']['AP15']
            extras_AP25 = results['bbox_3D']['AP25']
            extras_AP50 = results['bbox_3D']['AP50']
            extras_APn = results['bbox_3D']['APn']
            extras_APm = results['bbox_3D']['APm']
            extras_APf = results['bbox_3D']['APf']

        self.results_analysis[dataset_name] = {
            "iters": self.iter_label, 
            "AP2D": general_2D, "AP3D": general_3D, 
            "AP3D@15": extras_AP15, "AP3D@25": extras_AP25, "AP3D@50": extras_AP50, 
            "AP3D-N": extras_APn, "AP3D-M": extras_APm, "AP3D-F": extras_APf
        }

        # Performance per category
        results_cat = OrderedDict()
        for cat in category_names:
            cat_2D, cat_3D = (np.nan,) * 2
            if 'AP-{}'.format(cat) in results['bbox_2D']:
                cat_2D = results['bbox_2D']['AP-{}'.format(cat)]
                if not self.only_2d:
                    cat_3D = results['bbox_3D']['AP-{}'.format(cat)]
            if not np.isnan(cat_2D) or not np.isnan(cat_3D):
                results_cat[cat] = {"AP2D": cat_2D, "AP3D": cat_3D}
        utils_logperf.print_ap_category_histogram(dataset_name, results_cat)

    def summarize_all(self,):
        '''
        Report collective metrics when possible for the the Omni3D dataset.
        This uses pre-computed evaluation results from each dataset, 
        which were aggregated and cached while evaluating individually. 
        This process simply re-accumulate and summarizes them. 
        '''

        # First, double check that we have all the evaluations
        for dataset_name in self.dataset_names:
            if not dataset_name in self.results:
                self.evaluate(dataset_name)

        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        catId2contiguous = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        ordered_things = [thing_classes[catId2contiguous[cid]] for cid in self.overall_catIds]
        categories = set(ordered_things)

        evaluator2D = Omni3Deval(mode='2D')
        evaluator2D.params.catIds = list(self.overall_catIds)
        evaluator2D.params.imgIds = list(self.overall_imgIds)
        evaluator2D.evalImgs = True
        evaluator2D.evals_per_cat_area = self.evals_per_cat_area2D
        evaluator2D._paramsEval = copy.deepcopy(evaluator2D.params)
        evaluator2D.accumulate()
        summarize_str2D = evaluator2D.summarize()
        
        precisions = evaluator2D.eval['precision']

        metrics = ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl"]

        results2D = {
            metric: float(
                evaluator2D.stats[idx] * 100 if evaluator2D.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }

        for idx, name in enumerate(ordered_things):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results2D.update({"AP-" + "{}".format(name): float(ap * 100)})

        evaluator3D = Omni3Deval(mode='3D')
        evaluator3D.params.catIds = list(self.overall_catIds)
        evaluator3D.params.imgIds = list(self.overall_imgIds)
        evaluator3D.evalImgs = True
        evaluator3D.evals_per_cat_area = self.evals_per_cat_area3D
        evaluator3D._paramsEval = copy.deepcopy(evaluator3D.params)
        evaluator3D.accumulate()
        summarize_str3D = evaluator3D.summarize()
        
        precisions = evaluator3D.eval['precision']

        metrics = ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf"]

        results3D = {
            metric: float(
                evaluator3D.stats[idx] * 100 if evaluator3D.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }

        for idx, name in enumerate(ordered_things):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results3D.update({"AP-" + "{}".format(name): float(ap * 100)})


        # All concat categories
        general_2D, general_3D = (np.nan,) * 2

        general_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in categories])
        if not self.only_2d:
            general_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in categories])

        # Analysis performance
        extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf = (np.nan,) * 6
        if not self.only_2d:
            extras_AP15 = results3D['AP15']
            extras_AP25 = results3D['AP25']
            extras_AP50 = results3D['AP50']
            extras_APn = results3D['APn']
            extras_APm = results3D['APm']
            extras_APf = results3D['APf']

        self.results_analysis["<Concat>"] = {
            "iters": self.iter_label, 
            "AP2D": general_2D, "AP3D": general_3D, 
            "AP3D@15": extras_AP15, "AP3D@25": extras_AP25, "AP3D@50": extras_AP50, 
            "AP3D-N": extras_APn, "AP3D-M": extras_APm, "AP3D-F": extras_APf
        }

        # Omni3D Outdoor performance
        omni_2D, omni_3D = (np.nan,) * 2

        omni3d_outdoor_categories = get_omni3d_categories("omni3d_out")
        if len(omni3d_outdoor_categories - categories) == 0:
            omni_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in omni3d_outdoor_categories])
            if not self.only_2d:
                omni_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in omni3d_outdoor_categories])

        self.results_omni3d["Omni3D_Out"] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D}

        # Omni3D Indoor performance
        omni_2D, omni_3D = (np.nan,) * 2

        omni3d_indoor_categories = get_omni3d_categories("omni3d_in")
        if len(omni3d_indoor_categories - categories) == 0:
            omni_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in omni3d_indoor_categories])
            if not self.only_2d:
                omni_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in omni3d_indoor_categories])

        self.results_omni3d["Omni3D_In"] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D}

        # Omni3D performance
        omni_2D, omni_3D = (np.nan,) * 2

        omni3d_categories = get_omni3d_categories("omni3d")
        if len(omni3d_categories - categories) == 0:
            omni_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in omni3d_categories])
            if not self.only_2d:
                omni_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in omni3d_categories])

        self.results_omni3d["Omni3D"] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D}

        # Per-category performance for the cumulative datasets
        results_cat = OrderedDict()
        for cat in self.filter_settings['category_names']:
            cat_2D, cat_3D = (np.nan,) * 2
            if 'AP-{}'.format(cat) in results2D:
                cat_2D = results2D['AP-{}'.format(cat)]
                if not self.only_2d:
                    cat_3D = results3D['AP-{}'.format(cat)]
            if not np.isnan(cat_2D) or not np.isnan(cat_3D):
                results_cat[cat] = {"AP2D": cat_2D, "AP3D": cat_3D}
        
        utils_logperf.print_ap_category_histogram("<Concat>", results_cat)
        utils_logperf.print_ap_analysis_histogram(self.results_analysis)
        utils_logperf.print_ap_omni_histogram(self.results_omni3d)


def inference_on_dataset(model, data_loader):
    """
    Run model on the data_loader. 
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    
    num_devices = get_world_size()
    distributed = num_devices > 1
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    inference_json = []

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()

            for input, output in zip(inputs, outputs):

                prediction = {
                    "image_id": input["image_id"],
                    "K": input["K"],
                    "width": input["width"],
                    "height": input["height"],
                }

                # convert to json format
                instances = output["instances"].to('cpu')
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

                # store in overall predictions
                inference_json.append(prediction)

            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    if distributed:
        comm.synchronize()
        inference_json = comm.gather(inference_json, dst=0)
        inference_json = list(itertools.chain(*inference_json))

        if not comm.is_main_process():
            return []

    return inference_json

class Omni3DEvaluator(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=False,
        eval_prox=False,
        only_2d=False,
        filter_settings={},
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. For now, support only for "bbox".
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                    contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            eval_prox (bool): whether to perform proximity evaluation. For datasets that are not
                exhaustively annotated.
            only_2d (bool): evaluates only 2D performance if set to True
            filter_settions: settings for the dataset loader. TBD
        """

        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl
        self._eval_prox = eval_prox
        self._only_2d = only_2d
        self._filter_settings = filter_settings

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]

        else:
            max_dets_per_image = [1, 10, max_dets_per_image]

        self._max_dets_per_image = max_dets_per_image

        self._tasks = tasks
        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._omni_api = Omni3D([json_file], filter_settings)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._omni_api.dataset

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        # Optional image keys to keep when available
        img_keys_optional = ["p2"]

        for input, output in zip(inputs, outputs):

            prediction = {
                "image_id": input["image_id"],
                "K": input["K"],
                "width": input["width"],
                "height": input["height"],
            }

            # store optional keys when available
            for img_key in img_keys_optional:
                if img_key in input:
                    prediction.update({img_key: input[img_key]})

            # already in COCO format
            if type(output["instances"]) == list:
                prediction["instances"] = output["instances"]

            # tensor instances format
            else: 
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"]
                )

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _derive_omni_results(self, omni_eval, iou_type, mode, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            omni_eval (None or Omni3Deval): None represents no predictions from model.
            iou_type (str):
            mode (str): either "2D" or "3D"
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """
        assert mode in ["2D", "3D"]

        metrics = {
            "2D": ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl"],
            "3D": ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf"],
        }[mode]

        if iou_type != "bbox":
            raise ValueError("Support only for bbox evaluation.")

        if omni_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                omni_eval.stats[idx] * 100 if omni_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {} in {} mode: \n".format(iou_type, mode)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = omni_eval.eval["precision"]

        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_table = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_table,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info(
            "Per-category {} AP in {} mode: \n".format(iou_type, mode) + table
        )
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        omni_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(omni_results)

        omni3d_global_categories = MetadataCatalog.get('omni3d_model').thing_classes

        # the dataset results will store only the categories that are present
        # in the corresponding dataset, all others will be dropped. 
        dataset_results = []
        
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = (
                self._metadata.thing_dataset_id_to_contiguous_id
            )
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert (
                min(all_contiguous_ids) == 0
                and max(all_contiguous_ids) == num_classes - 1
            )

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in omni_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

                cat_name = omni3d_global_categories[category_id]

                if cat_name in self._metadata.thing_classes:
                    dataset_results.append(result)

        # replace the results with the filtered
        # instances that are in vocabulary. 
        omni_results = dataset_results

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "omni_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(omni_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox"}, f"Got unknown task: {task}!"
            evals, log_strs = (
                _evaluate_predictions_on_omni(
                    self._omni_api,
                    omni_results,
                    task,
                    img_ids=img_ids,
                    only_2d=self._only_2d,
                    eval_prox=self._eval_prox,
                )
                if len(omni_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            modes = evals.keys()
            for mode in modes:
                res = self._derive_omni_results(
                    evals[mode],
                    task,
                    mode,
                    class_names=self._metadata.get("thing_classes"),
                )
                self._results[task + "_" + format(mode)] = res
                self._results[task + "_" + format(mode) + '_evalImgs'] = evals[mode].evalImgs
                self._results[task + "_" + format(mode) + '_evals_per_cat_area'] = evals[mode].evals_per_cat_area

            self._results["log_str_2D"] = log_strs["2D"]
            
            if "3D" in log_strs:
                self._results["log_str_3D"] = log_strs["3D"]


def _evaluate_predictions_on_omni(
    omni_gt,
    omni_results,
    iou_type,
    img_ids=None,
    only_2d=False,
    eval_prox=False,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(omni_results) > 0
    log_strs, evals = {}, {}

    omni_dt = omni_gt.loadRes(omni_results)

    modes = ["2D"] if only_2d else ["2D", "3D"]

    for mode in modes:
        omni_eval = Omni3Deval(
            omni_gt, omni_dt, iouType=iou_type, mode=mode, eval_prox=eval_prox
        )
        if img_ids is not None:
            omni_eval.params.imgIds = img_ids

        omni_eval.evaluate()
        omni_eval.accumulate()
        log_str = omni_eval.summarize()
        log_strs[mode] = log_str
        evals[mode] = omni_eval

    return evals, log_strs


def instances_to_coco_json(instances, img_id):

    num_instances = len(instances)

    if num_instances == 0:
        return []

    boxes = BoxMode.convert(
        instances.pred_boxes.tensor.numpy(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    ).tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    if hasattr(instances, "pred_bbox3D"):
        bbox3D = instances.pred_bbox3D.tolist()
        center_cam = instances.pred_center_cam.tolist()
        center_2D = instances.pred_center_2D.tolist()
        dimensions = instances.pred_dimensions.tolist()
        pose = instances.pred_pose.tolist()
    else:
        # dummy
        bbox3D = np.ones([num_instances, 8, 3]).tolist()
        center_cam = np.ones([num_instances, 3]).tolist()
        center_2D = np.ones([num_instances, 2]).tolist()
        dimensions = np.ones([num_instances, 3]).tolist()
        pose = np.ones([num_instances, 3, 3]).tolist()

    results = []
    for k in range(num_instances):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "depth": np.array(bbox3D[k])[:, 2].mean(),
            "bbox3D": bbox3D[k],
            "center_cam": center_cam[k],
            "center_2D": center_2D[k],
            "dimensions": dimensions[k],
            "pose": pose[k],
        }

        results.append(result)
    return results


# ---------------------------------------------------------------------
#                               Omni3DParams
# ---------------------------------------------------------------------
class Omni3DParams:
    """
    Params for the Omni evaluation API
    """

    def setDet2DParams(self):
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]

        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setDet3DParams(self):
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.05, 0.5, int(np.round((0.5 - 0.05) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [[0, 1e5], [0, 10], [10, 35], [35, 1e5]]
        self.areaRngLbl = ["all", "near", "medium", "far"]
        self.useCats = 1

    def __init__(self, mode="2D"):
        """
        Args:
            iouType (str): defines 2D or 3D evaluation parameters.
                One of {"2D", "3D"}
        """

        if mode == "2D":
            self.setDet2DParams()

        elif mode == "3D":
            self.setDet3DParams()

        else:
            raise Exception("mode %s not supported" % (mode))

        self.iouType = "bbox"
        self.mode = mode
        # the proximity threshold defines the neighborhood
        # when evaluating on non-exhaustively annotated datasets
        self.proximity_thresh = 0.3


# ---------------------------------------------------------------------
#                               Omni3Deval
# ---------------------------------------------------------------------
class Omni3Deval(COCOeval):
    """
    Wraps COCOeval for 2D or 3D box evaluation depending on mode
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="bbox", mode="2D", eval_prox=False
    ):
        """
        Initialize COCOeval using coco APIs for Gt and Dt
        Args:
            cocoGt: COCO object with ground truth annotations
            cocoDt: COCO object with detection results
            iouType: (str) defines the evaluation type. Supports only "bbox" now.
            mode: (str) defines whether to evaluate 2D or 3D performance.
                One of {"2D", "3D"}
            eval_prox: (bool) if True, performs "Proximity Evaluation", i.e.
                evaluates detections in the proximity of the ground truth2D boxes.
                This is used for datasets which are not exhaustively annotated.
        """
        if not iouType:
            print("iouType not specified. use default iouType bbox")
        elif iouType != "bbox":
            print("no support for %s iouType" % (iouType))
        self.mode = mode
        if mode not in ["2D", "3D"]:
            raise Exception("mode %s not supported" % (mode))
        self.eval_prox = eval_prox
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list) 

        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Omni3DParams(mode)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts

        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.evals_per_cat_area = None

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        """
        
        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        ignore_flag = "ignore2D" if self.mode == "2D" else "ignore3D"
        for gt in gts:
            gt[ignore_flag] = gt[ignore_flag] if ignore_flag in gt else 0

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''

        print('Accumulating evaluation results...')
        assert self.evalImgs, 'Please run evaluate() first'

        tic = time.time()

        # allows input customized parameters
        if p is None:
            p = self.params

        p.catIds = p.catIds if p.useCats == 1 else [-1]

        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)

        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval

        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)

        # get inds to evaluate
        catid_list = [k for n, k in enumerate(p.catIds)  if k in setK]
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]

        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        has_precomputed_evals = not (self.evals_per_cat_area is None)
        
        if has_precomputed_evals:
            evals_per_cat_area = self.evals_per_cat_area
        else:
            evals_per_cat_area = {}

        # retrieve E at each category, area range, and max number of detections
        for k, (k0, catId) in enumerate(zip(k_list, catid_list)):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0

                if has_precomputed_evals:
                    E = evals_per_cat_area[(catId, a)]

                else:
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    evals_per_cat_area[(catId, a)] = E

                if len(E) == 0:
                    continue

                for m, maxDet in enumerate(m_list):

                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0)

                    if npig == 0:
                        continue

                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]

                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass

                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)

        self.evals_per_cat_area = evals_per_cat_area

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        """

        print("Running per image evaluation...")
        
        p = self.params
        print("Evaluate annotation type *{}*".format(p.iouType))

        tic = time.time()

        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))

        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        
        catIds = p.catIds if p.useCats else [-1]

        # loop through images, area range, max detection number
        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]

        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId):
        """
        ComputeIoU computes the IoUs by sorting based on "score"
        for either 2D boxes (in 2D mode) or 3D boxes (in 3D mode)
        """
        
        device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]

        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "bbox":
            if self.mode == "2D":
                g = [g["bbox"] for g in gt]
                d = [d["bbox"] for d in dt]
            elif self.mode == "3D":
                g = [g["bbox3D"] for g in gt]
                d = [d["bbox3D"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        # iscrowd is required in builtin maskUtils so we
        # use a dummy buffer for it
        iscrowd = [0 for o in gt]
        if self.mode == "2D":
            ious = maskUtils.iou(d, g, iscrowd)

        elif len(d) > 0 and len(g) > 0:
            
            # For 3D eval, we want to run IoU in CUDA if available
            if torch.cuda.is_available() and len(d) * len(g) < MAX_DTS_CROSS_GTS_FOR_IOU3D:
                device = torch.device("cuda:0") 
            else:
                device = torch.device("cpu")
            
            dd = torch.tensor(d, device=device, dtype=torch.float32)
            gg = torch.tensor(g, device=device, dtype=torch.float32)

            ious = box3d_overlap(dd, gg).cpu().numpy()

        else:
            ious = []

        in_prox = None

        if self.eval_prox:
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
            iscrowd = [0 for o in gt]
            ious2d = maskUtils.iou(d, g, iscrowd)

            if type(ious2d) == list:
                in_prox = []

            else:
                in_prox = ious2d > p.proximity_thresh
        
        return ious, in_prox

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        Perform evaluation for single category and image
        Returns:
            dict (single image results)
        """

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]

        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        flag_range = "area" if self.mode == "2D" else "depth"
        flag_ignore = "ignore2D" if self.mode == "2D" else "ignore3D"

        for g in gt:
            if g[flag_ignore] or (g[flag_range] < aRng[0] or g[flag_range] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]

        # load computed ious
        ious = (
            self.ious[imgId, catId][0][:, gtind]
            if len(self.ious[imgId, catId][0]) > 0
            else self.ious[imgId, catId][0]
        )

        if self.eval_prox:
            in_prox = (
                self.ious[imgId, catId][1][:, gtind]
                if len(self.ious[imgId, catId][1]) > 0
                else self.ious[imgId, catId][1]
            )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):

                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1

                    for gind, g in enumerate(gt):
                        # in case of proximity evaluation, if not in proximity continue
                        if self.eval_prox and not in_prox[dind, gind]:
                            continue

                        # if this gt already matched, continue
                        if gtm[tind, gind] > 0:
                            continue

                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break

                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue

                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind

                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue

                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        # set unmatched detections outside of area range to ignore
        a = np.array(
            [d[flag_range] < aRng[0] or d[flag_range] > aRng[1] for d in dt]
        ).reshape((1, len(dt)))

        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

        # in case of proximity evaluation, ignore detections which are far from gt regions
        if self.eval_prox and len(in_prox) > 0:
            dt_far = in_prox.any(1) == 0
            dtIg = np.logical_or(dtIg, np.repeat(dt_far.reshape((1, len(dt))), T, 0))

        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(mode, ap=1, iouThr=None, areaRng="all", maxDets=100, log_str=""):
            p = self.params
            eval = self.eval

            if mode == "2D":
                iStr = (" {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}")

            elif mode == "3D":
                iStr = " {:<18} {} @[ IoU={:<9} | depth={:>6s} | maxDets={:>3d} ] = {:0.3f}"

            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"

            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:

                # dimension of precision: [TxRxKxAxM]
                s = eval["precision"]

                # IoU
                if iouThr is not None:
                    t = np.where(np.isclose(iouThr, p.iouThrs.astype(float)))[0]
                    s = s[t]

                s = s[:, :, :, aind, mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
                
            else:
                mean_s = np.mean(s[s > -1])

            if log_str != "":
                log_str += "\n"

            log_str += "mode={} ".format(mode) + \
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            
            return mean_s, log_str

        def _summarizeDets(mode):

            params = self.params

            # the thresholds here, define the thresholds printed in `derive_omni_results`
            thres = [0.5, 0.75, 0.95] if mode == "2D" else [0.15, 0.25, 0.50]

            stats = np.zeros((13,))
            stats[0], log_str = _summarize(mode, 1)

            stats[1], log_str = _summarize(
                mode, 1, iouThr=thres[0], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[2], log_str = _summarize(
                mode, 1, iouThr=thres[1], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[3], log_str = _summarize(
                mode, 1, iouThr=thres[2], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[4], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[5], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[6], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[3],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[7], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[0], log_str=log_str
            )

            stats[8], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[1], log_str=log_str
            )

            stats[9], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[2], log_str=log_str
            )

            stats[10], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[11], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[12], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[3],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )
            
            return stats, log_str

        if not self.eval:
            raise Exception("Please run accumulate() first")

        stats, log_str = _summarizeDets(self.mode)
        self.stats = stats

        return log_str