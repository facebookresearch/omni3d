# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import sys
import numpy as np
import copy
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    inference_on_dataset,
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    load_omni3d_json,
    DatasetMapper3D,
    build_detection_train_loader,
    build_detection_test_loader
)
from cubercnn.evaluation import (
    Omni3DEvaluator, OmniEval,
)
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis, data

OMNI3D_OFFICIAL = {'chair', 'table', 'cabinet', 'car', 'lamp', 'books', 'sofa', 'pedestrian', 'picture', 'window', 'pillow', 'truck', 'door', 'blinds', 'sink', 'shelves', 'television', 'shoes', 'cup', 'bottle', 'bookcase', 'laptop', 'desk', 'cereal box', 'floor mat', 'traffic cone', 'mirror', 'barrier', 'counter', 'camera', 'bicycle', 'toilet', 'bus', 'bed', 'refrigerator', 'trailer', 'box', 'oven', 'clothes', 'van', 'towel', 'motorcycle', 'night stand', 'stove', 'machine', 'stationery', 'bathtub', 'cyclist', 'curtain', 'bin'}
INDOOR_COMMON = {'table', 'bed', 'sofa', 'television', 'bathtub', 'sink', 'shelves', 'cabinet', 'refrigerator', 'chair'}
OUTDOOR_COMMON = {'truck', 'pedestrian', 'car'}

MAX_TRAINING_ATTEMPTS = 10

def do_test(cfg, model, iteration='final', storage=None):
    
    results = OrderedDict()
    
    filter_settings = data.get_filter_settings_from_cfg(cfg)    
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    overall_imgIds = set()
    overall_catIds = set()

    overall_extras_str = '\n--------------------------------------------- Detailed ----------------------------------------------'
    overall_extras_str += '\n{:20}  {} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format('datasets', 'iter', 'AP_2D', 'AP_3D', 'AP15_3D', 'AP25_3D', 'AP50_3D', 'APn_3D', 'APm_3D', 'APf_3D')

    overall_log_str    = '\n--------------------------------------------- Overview ----------------------------------------------'
    overall_log_str += '\n{:20}  {} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format('datasets', 'iter', 'AP_2D', 'AP_3D', 'In_2D', 'In_3D', 'Out_2D', 'Out_3D', 'Omni_2D', 'Omni_3D')
    
    # These store the evaluations for each category and area,
    # concatenated from ALL evaluated datasets. Doing so avoids
    # the need to re-compute them when accumulating results.
    evals_per_cat_area2D = {}
    evals_per_cat_area3D = {}
    
    datasets_test = cfg.DATASETS.TEST

    for dataset_name in datasets_test:

        '''
        Cycle through each dataset and test them individually.
        This loop keeps track of each per-image evaluation result, 
        so that it doesn't need to be re-computed for the collective.
        '''

        data_loader = build_detection_test_loader(cfg, dataset_name)
        only_2d=cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
        output_folder =  os.path.join(cfg.OUTPUT_DIR, "inference", 'iter_{}'.format(iteration), dataset_name)
        evaluator = Omni3DEvaluator(
            dataset_name, output_dir=output_folder, 
            filter_settings=filter_settings, only_2d=only_2d, 
            eval_prox=('Objectron' in dataset_name or 'SUNRGBD' in dataset_name)
        )

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i

        if comm.is_main_process():
            
            overall_imgIds.update(set(evaluator._omni_api.getImgIds()))
            overall_catIds.update(set(evaluator._omni_api.getCatIds()))

            logger.info('\n'+results_i['log_str_2D'].replace('mode=2D', '{} iter={} mode=2D'.format(dataset_name, iteration)))
            
            # store the partially accumulated evaluations per category per area
            for key, item in results_i['bbox_2D_evals_per_cat_area'].items():
                if not key in evals_per_cat_area2D:
                    evals_per_cat_area2D[key] = []
                evals_per_cat_area2D[key] += item

            if not only_2d:
                # store the partially accumulated evaluations per category per area
                for key, item in results_i['bbox_3D_evals_per_cat_area'].items():
                    if not key in evals_per_cat_area3D:
                        evals_per_cat_area3D[key] = []
                    evals_per_cat_area3D[key] += item

                logger.info('\n'+results_i['log_str_3D'].replace('mode=3D', '{} iter={} mode=3D'.format(dataset_name, iteration)))

            categories_in_common = set(cat for cat in cfg.DATASETS.CATEGORY_NAMES if 'AP-{}'.format(cat) in results_i['bbox_2D'])

            # default are all NaN, assuming the categories are not available.
            general_2D, general_3D, outdoor_common_2D, outdoor_common_3D, indoor_common_2D, indoor_common_3D, omni_common_2D, omni_common_3D = (np.nan,)*8

            general_2D = np.mean([results_i['bbox_2D']['AP-{}'.format(cat)] for cat in categories_in_common])

            if not only_2d:
                general_3D = np.mean([results_i['bbox_3D']['AP-{}'.format(cat)] for cat in categories_in_common])

            # valid omni common
            if len(OMNI3D_OFFICIAL - categories_in_common) == 0:
                omni_common_2D = np.mean([results_i['bbox_2D']['AP-{}'.format(cat)] for cat in OMNI3D_OFFICIAL])
                
                if not only_2d:
                    omni_common_3D = np.mean([results_i['bbox_3D']['AP-{}'.format(cat)] for cat in OMNI3D_OFFICIAL])

            # valid outdoor common
            if len(OUTDOOR_COMMON - categories_in_common) == 0:
                outdoor_common_2D = np.mean([results_i['bbox_2D']['AP-{}'.format(cat)] for cat in OUTDOOR_COMMON])
                
                if not only_2d:
                    outdoor_common_3D = np.mean([results_i['bbox_3D']['AP-{}'.format(cat)] for cat in OUTDOOR_COMMON])
                
            # valid indoor common
            if len(INDOOR_COMMON - categories_in_common) == 0:
                indoor_common_2D = np.mean([results_i['bbox_2D']['AP-{}'.format(cat)] for cat in INDOOR_COMMON])
                
                if not only_2d:
                    indoor_common_3D = np.mean([results_i['bbox_3D']['AP-{}'.format(cat)] for cat in INDOOR_COMMON])

            extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf = (np.nan,)*6

            if not only_2d:
                extras_AP15 = results_i['bbox_3D']['AP15']
                extras_AP25 = results_i['bbox_3D']['AP25']
                extras_AP50 = results_i['bbox_3D']['AP50']
                extras_APn = results_i['bbox_3D']['APn']
                extras_APm = results_i['bbox_3D']['APm']
                extras_APf = results_i['bbox_3D']['APf']
            
            overall_extras_str += '\n{:20}  {} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(
                dataset_name, iteration, general_2D, general_3D, extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf
            )

            overall_log_str += '\n{:20}  {} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(
                dataset_name, iteration, general_2D, general_3D, indoor_common_2D, indoor_common_3D, outdoor_common_2D, outdoor_common_3D, omni_common_2D, omni_common_3D
            )

            results_str = '\n---------------------------------------------------------------------------'
            results_str += '\n{}  {}   {:15} {:>15} {:>15}'.format(dataset_name, 'iter', '', 'AP_2D', 'AP_3D')
            for cat in cfg.DATASETS.CATEGORY_NAMES:
                if 'AP-{}'.format(cat) in results_i['bbox_2D']:
                    results_str += '\n{}  {}   {:15} {:15.4f} {:15.4f}'.format(dataset_name, iteration, cat, results_i['bbox_2D']['AP-{}'.format(cat)], 0 if only_2d else results_i['bbox_3D']['AP-{}'.format(cat)])
            results_str += '\n---------------------------------------------------------------------------'

            logger.info(results_str)
        
        if comm.is_main_process():
            
            '''
            Visualize some predictions from the instances. 
            '''

            detections = torch.load(os.path.join(output_folder, 'instances_predictions.pth'))
            log_str = vis.visualize_from_instances(
                detections, data_loader.dataset, dataset_name, 
                cfg.INPUT.MIN_SIZE_TEST, output_folder, MetadataCatalog.get('omni3d_model').thing_classes, iteration
            )
            logger.info(log_str)

    if comm.is_main_process():

        
        '''
        Report collective metrics when possible for the the Omni3D dataset.
        This uses pre-computed evaluation results from each dataset, which were
        aggregated and cached while evaluating them individually. 
        This process simply re-accumulate and summarizes them. 
        '''

        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        catId2contiguous = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        ordered_things = [thing_classes[catId2contiguous[cid]] for cid in overall_catIds]
        categories_in_common = set(ordered_things)

        evaluator2D = OmniEval(mode='2D')
        evaluator2D.params.catIds = list(overall_catIds)
        evaluator2D.params.imgIds = list(overall_imgIds)
        evaluator2D.evalImgs = True
        evaluator2D.evals_per_cat_area = evals_per_cat_area2D
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

        evaluator3D = OmniEval(mode='3D')
        evaluator3D.params.catIds = list(overall_catIds)
        evaluator3D.params.imgIds = list(overall_imgIds)
        evaluator3D.evalImgs = True
        evaluator3D.evals_per_cat_area = evals_per_cat_area3D
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

        general_2D, general_3D, outdoor_common_2D, outdoor_common_3D, indoor_common_2D, indoor_common_3D, omni_common_2D, omni_common_3D = (np.nan,)*8

        general_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in categories_in_common])

        if not only_2d:
            general_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in categories_in_common])

        # valid omni common
        if len(OMNI3D_OFFICIAL - categories_in_common) == 0:
            omni_common_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in OMNI3D_OFFICIAL])
            
            if not only_2d:
                omni_common_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in OMNI3D_OFFICIAL])

        # valid outdoor common
        if len(OUTDOOR_COMMON - categories_in_common) == 0:
            outdoor_common_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in OUTDOOR_COMMON])
            
            if not only_2d:
                outdoor_common_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in OUTDOOR_COMMON])
            
        # valid indoor common
        if len(INDOOR_COMMON - categories_in_common) == 0:
            indoor_common_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in INDOOR_COMMON])
            
            if not only_2d:
                indoor_common_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in INDOOR_COMMON])

        dataset_name = '<Concat>'
        results_str = '\n---------------------------------------------------------------------------'
        results_str += '\n{}  {}   {:15} {:>15} {:>15}'.format(dataset_name, 'iter', '', 'AP_2D', 'AP_3D')
        for cat in categories_in_common:
            if 'AP-{}'.format(cat) in results2D:
                results_str += '\n{}  {}   {:15} {:15.4f} {:15.4f}'.format(dataset_name, iteration, cat, results2D['AP-{}'.format(cat)], 0 if only_2d else results3D['AP-{}'.format(cat)])
        results_str += '\n---------------------------------------------------------------------------'

        logger.info(results_str)

        extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf = (np.nan,)*6

        if not only_2d:
            extras_AP15 = results3D['AP15']
            extras_AP25 = results3D['AP25']
            extras_AP50 = results3D['AP50']
            extras_APn = results3D['APn']
            extras_APm = results3D['APm']
            extras_APf = results3D['APf']
         
        overall_extras_str += '\n{:20}  {} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(
            dataset_name, iteration, general_2D, general_3D, extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf
        )
        overall_extras_str += '\n-----------------------------------------------------------------------------------------------------'
        overall_log_str += '\n{:20}  {} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(
            dataset_name, iteration, general_2D, general_3D, indoor_common_2D, indoor_common_3D, outdoor_common_2D, outdoor_common_3D, omni_common_2D, omni_common_3D
        )
        overall_log_str += '\n-----------------------------------------------------------------------------------------------------'
        logger.info(overall_extras_str)
        logger.info(overall_log_str)


def do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=False):

    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # bookkeeping
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)    
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    
    # create the dataloader
    data_mapper = DatasetMapper3D(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src)

    # give the mapper access to dataset_ids
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

    if cfg.MODEL.WEIGHTS_PRETRAIN != '':
        
        # load ONLY the model, no checkpointables.
        checkpointer.load(cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    # determine the starting iteration, if resuming
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))

    if not cfg.MODEL.USE_BN:
        freeze_bn(model)

    world_size = comm.get_world_size()

    # if the loss diverges for more than the below TOLERANCE
    # as a percent of the iterations, the training will stop.
    # This is only enabled if "STABILIZE" is on, which 
    # prevents a single example from exploding the training. 
    iterations_success = 0
    iterations_explode = 0
    
    # when loss > recent_loss * TOLERANCE, then it could be a
    # diverging/failing model, which we should skip all updates for.
    TOLERANCE = 4.0         

    GAMMA = 0.02            # rolling average weight gain
    recent_loss = None      # stores the most recent loss magnitude

    data_iter = iter(data_loader)

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    named_params = list(model.named_parameters())

    with EventStorage(start_iter) as storage:
        
        while True:

            data = next(data_iter)
            storage.iter = iteration

            # forward
            loss_dict = model(data)
            losses = sum(loss_dict.values())

            # reduce
            loss_dict_reduced = {k: v.item() for k, v in allreduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
            # sync up
            comm.synchronize()

            if recent_loss is None:

                # init recent loss fairly high
                recent_loss = losses_reduced*2.0

            # Is stabilization enabled, and loss high or NaN?
            diverging_model = cfg.MODEL.STABILIZE > 0 and \
                        (losses_reduced > recent_loss*TOLERANCE or \
                            not (np.isfinite(losses_reduced)) or np.isnan(losses_reduced))

            if diverging_model:
                # clip and warn the user.
                losses = losses.clip(0, 1) 
                logger.warning('Skipping gradient update due to higher than normal loss {:.2f} vs. rolling mean {:.2f}, Dict-> {}'.format(
                    losses_reduced, recent_loss, loss_dict_reduced
                ))
            else:
                # compute rolling average of loss
                recent_loss = recent_loss * (1-GAMMA) + losses_reduced*GAMMA
            
            if comm.is_main_process():
                # send loss scalars to tensorboard.
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        
            # backward and step
            optimizer.zero_grad()
            losses.backward()

            # if the loss is not too high, 
            # we still want to check gradients.
            if not diverging_model:

                if cfg.MODEL.STABILIZE > 0:
                    
                    for name, param in named_params:

                        if param.grad is not None:
                            diverging_model = torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                        
                        if diverging_model:
                            logger.warning('Skipping gradient update due to inf/nan detection, loss is {}'.format(loss_dict_reduced))
                            break

            # convert exploded to a float, then allreduce it, 
            # if any process gradients have exploded then we skip together.
            diverging_model = torch.tensor(float(diverging_model)).cuda()

            if world_size > 1:
                dist.all_reduce(diverging_model)

            # sync up
            comm.synchronize()

            if diverging_model > 0:
                optimizer.zero_grad()
                iterations_explode += 1

            else:
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                iterations_success += 1

            total_iterations = iterations_success + iterations_explode

            # Only retry if we have trained sufficiently long relative
            # to the latest checkpoint, which we would otherwise revert back to.
            retry = (iterations_explode / total_iterations) >= cfg.MODEL.STABILIZE \
                    and (total_iterations > cfg.SOLVER.CHECKPOINT_PERIOD*1/2)
            
            # Important for dist training. Convert to a float, then allreduce it, 
            # if any process gradients have exploded then we must skip together.
            retry = torch.tensor(float(retry)).cuda()
            
            if world_size > 1:
                dist.all_reduce(retry)

            # sync up
            comm.synchronize()

            # any processes need to retry
            if retry > 0:

                # instead of failing, try to resume the iteration instead. 
                logger.warning('!! Restarting training at {} iters. Exploding loss {:d}% of iters !!'.format(
                    iteration, int(100*(iterations_explode / (iterations_success + iterations_explode)))
                ))

                # send these to garbage, for ideally a cleaner restart.
                del data_mapper
                del data_loader
                del optimizer
                del checkpointer
                del periodic_checkpointer
                return False
                
            scheduler.step()

            # Evaluate only when the loss is not diverging.
            if not (diverging_model > 0) and \
                (do_eval and ((iteration + 1) % cfg.TEST.EVAL_PERIOD) == 0 and iteration != (max_iter - 1)):

                logger.info('Starting test for iteration {}'.format(iteration+1))
                do_test(cfg, model, iteration=iteration+1, storage=storage)
                comm.synchronize()
                
                if not cfg.MODEL.USE_BN: 
                    freeze_bn(model)

            # Flush events
            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            
            # Do not bother checkpointing if there is potential for a diverging model.
            if not (diverging_model > 0) and \
                (iterations_explode / total_iterations) < 0.5*cfg.MODEL.STABILIZE:
                periodic_checkpointer.step(iteration)

            iteration += 1

            if iteration >= max_iter:
                break
    
    # success
    return True

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file
    
    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(cfg, dataset_name, filter_settings, filter_empty=True)
    
    datasets_test = cfg.DATASETS.TEST

    for dataset_name in datasets_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            simple_register(cfg, dataset_name, filter_settings, filter_empty=False)
    
    return cfg


def simple_register(cfg, dataset_name, filter_settings, filter_empty=False):

    path_to_json = os.path.join('datasets', 'Omni3D', dataset_name + '.json')
    path_to_image_root = 'datasets'

    DatasetCatalog.register(dataset_name, lambda: load_omni3d_json(
        path_to_json, path_to_image_root, 
        dataset_name, filter_settings, filter_empty=filter_empty
    ))

    MetadataCatalog.get(dataset_name).set(json_file=path_to_json, image_root=path_to_image_root, evaluator_type="coco")


def main(args):
    
    cfg = setup(args)

    logger.info('Preprocessing Training Datasets')

    filter_settings = data.get_filter_settings_from_cfg(cfg)

    priors = None

    if args.eval_only:
        category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        metadata = util.load_json(category_path)

        # register the categories
        thing_classes = metadata['thing_classes']
        id_map = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
        MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id  = id_map

    else: 

        # setup and join the data.
        dataset_paths = [os.path.join('datasets', 'Omni3D', name + '.json') for name in cfg.DATASETS.TRAIN]
        datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)

        # determine the meta data given the datasets used. 
        data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        
        '''
        It may be useful to keep track of which categories are annotated/known
        for each dataset in use, in case a method wants to use this information.
        '''

        infos = datasets.dataset['info']

        if type(infos) == dict:
            infos = [datasets.dataset['info']]

        dataset_id_to_unknown_cats = {}
        possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
        
        dataset_id_to_src = {}

        for info in infos:
            dataset_id = info['id']
            known_category_training_ids = set()

            if not dataset_id in dataset_id_to_src:
                dataset_id_to_src[dataset_id] = info['source']

            for id in info['known_category_ids']:
                if id in dataset_id_to_contiguous_id:
                    known_category_training_ids.add(dataset_id_to_contiguous_id[id])
            
            # determine and store the unknown categories.
            unknown_categories = possible_categories - known_category_training_ids
            dataset_id_to_unknown_cats[dataset_id] = unknown_categories

            # log the per-dataset categories
            logger.info('Available categories for {}'.format(info['name']))
            logger.info([thing_classes[i] for i in (possible_categories & known_category_training_ids)])

        # compute priors given the training data.
        priors = util.compute_priors(cfg, datasets)
    
    '''
    The training loops can attempt to train for N times.
    This catches a divergence or other failure modes. 
    '''

    remaining_attempts = MAX_TRAINING_ATTEMPTS
    while remaining_attempts > 0:

        # build the training model.
        model = build_model(cfg, priors=priors)

        if remaining_attempts == MAX_TRAINING_ATTEMPTS:
            # log the first attempt's settings.
            logger.info("Model:\n{}".format(model))

        if args.eval_only:
            # skip straight to eval mode
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            return do_test(cfg, model)

        # setup distributed training.
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], 
                broadcast_buffers=False, find_unused_parameters=True
            )

        # train full model, potentially with resume.
        if do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=args.resume):
            break
        else:

            # allow restart when a model fails to train.
            remaining_attempts -= 1
            del model

    if remaining_attempts == 0:
        # Exit if the model could not finish without diverging. 
        raise ValueError('Training failed')
        
    return do_test(cfg, model)

def allreduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = comm.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )