# Copyright (c) Meta Platforms, Inc. and affiliates
import json
import time
import os
import contextlib
import io
import logging
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict
from fvcore.common.timer import Timer
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from cubercnn import util

VERSION = '0.1'

logger = logging.getLogger(__name__)

def get_version():
    return VERSION

def get_global_dataset_stats(path_to_stats=None, reset=False):

    if path_to_stats is None:
        path_to_stats = os.path.join('datasets', 'Omni3D', 'stats.json')

    if os.path.exists(path_to_stats) and not reset:
        stats = util.load_json(path_to_stats)
    
    else:
        stats = {
            'n_datasets': 0,
            'n_ims': 0,
            'n_anns': 0,
            'categories': []
        }

    return stats


def save_global_dataset_stats(stats, path_to_stats=None):

    if path_to_stats is None:
        path_to_stats = os.path.join('datasets', 'Omni3D', 'stats.json')

    util.save_json(path_to_stats, stats)


def get_filter_settings_from_cfg(cfg=None):

    if cfg is None:
        return {
            'category_names': [], 
            'ignore_names': [], 
            'truncation_thres': 0.99, 
            'visibility_thres': 0.01,
            'min_height_thres': 0.00,
            'max_height_thres': 1.50,
            'modal_2D_boxes': False,
            'trunc_2D_boxes': False,
            'max_depth': 1e8,
        }
    else:
        return {
            'category_names': cfg.DATASETS.CATEGORY_NAMES, 
            'ignore_names': cfg.DATASETS.IGNORE_NAMES, 
            'truncation_thres': cfg.DATASETS.TRUNCATION_THRES, 
            'visibility_thres': cfg.DATASETS.VISIBILITY_THRES,
            'min_height_thres': cfg.DATASETS.MIN_HEIGHT_THRES,
            'modal_2D_boxes': cfg.DATASETS.MODAL_2D_BOXES,
            'trunc_2D_boxes': cfg.DATASETS.TRUNC_2D_BOXES,
            'max_depth': cfg.DATASETS.MAX_DEPTH,
            
            # TODO expose as a config
            'max_height_thres': 1.50,
        }


def is_ignore(anno, filter_settings, image_height):

    ignore = anno['behind_camera'] 
    ignore |= (not bool(anno['valid3D']))

    if ignore:
        return ignore

    ignore |= anno['dimensions'][0] <= 0
    ignore |= anno['dimensions'][1] <= 0
    ignore |= anno['dimensions'][2] <= 0
    ignore |= anno['center_cam'][2] > filter_settings['max_depth']
    ignore |= (anno['lidar_pts'] == 0)
    ignore |= (anno['segmentation_pts'] == 0)
    ignore |= (anno['depth_error'] > 0.5)
    
    # tightly annotated 2D boxes are not always available.
    if filter_settings['modal_2D_boxes'] and 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
        bbox2D =  BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    # truncated projected 2D boxes are also not always available.
    elif filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
        bbox2D =  BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    # use the projected 3D --> 2D box, which requires a visible 3D cuboid.
    elif 'bbox2D_proj' in anno:
        bbox2D =  BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    else:
        bbox2D = anno['bbox']

    ignore |= bbox2D[3] <= filter_settings['min_height_thres']*image_height
    ignore |= bbox2D[3] >= filter_settings['max_height_thres']*image_height
        
    ignore |= (anno['truncation'] >=0 and anno['truncation'] >= filter_settings['truncation_thres'])
    ignore |= (anno['visibility'] >= 0 and anno['visibility'] <= filter_settings['visibility_thres'])
    
    if 'ignore_names' in filter_settings:
        ignore |= anno['category_name'] in filter_settings['ignore_names']

    return ignore


def simple_register(dataset_name, filter_settings, filter_empty=False, datasets_root_path=None):

    if datasets_root_path is None:
        datasets_root_path = path_to_json = os.path.join('datasets', 'Omni3D',)
    
    path_to_json = os.path.join(datasets_root_path, dataset_name + '.json')
    path_to_image_root = 'datasets'

    DatasetCatalog.register(dataset_name, lambda: load_omni3d_json(
        path_to_json, path_to_image_root, 
        dataset_name, filter_settings, filter_empty=filter_empty
    ))

    MetadataCatalog.get(dataset_name).set(json_file=path_to_json, image_root=path_to_image_root, evaluator_type="coco")

class Omni3D(COCO):
    '''
    Class for COCO-like dataset object. Not inherently related to 
    use with Detectron2 or training per se. 
    '''

    def __init__(self, annotation_files, filter_settings=None):
             
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
       
        if isinstance(annotation_files, str):
            annotation_files = [annotation_files,]
        
        cats_ids_master = []
        cats_master = []
        
        for annotation_file in annotation_files:

            _, name, _ = util.file_parts(annotation_file)

            print('loading {} annotations into memory...'.format(name))
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))

            if type(dataset['info']) == list:
                dataset['info'] = dataset['info'][0]
                
            dataset['info']['known_category_ids'] = [cat['id'] for cat in dataset['categories']]

            # first dataset
            if len(self.dataset) == 0:
                self.dataset = dataset
            
            # concatenate datasets
            else:

                if type(self.dataset['info']) == dict:
                    self.dataset['info'] = [self.dataset['info']]
                    
                self.dataset['info'] += [dataset['info']]
                self.dataset['annotations'] += dataset['annotations']
                self.dataset['images'] += dataset['images']
            
            # sort through categories
            for cat in dataset['categories']:

                if not cat['id'] in cats_ids_master:
                    cats_ids_master.append(cat['id'])
                    cats_master.append(cat)

        if filter_settings is None:

            # include every category in the master list
            self.dataset['categories'] = [
                cats_master[i] 
                for i in np.argsort(cats_ids_master) 
            ]
            
        else:
        
            # determine which categories we may actually use for filtering.
            trainable_cats = set(filter_settings['ignore_names']) | set(filter_settings['category_names'])

            # category names are provided to us
            if len(filter_settings['category_names']) > 0:

                self.dataset['categories'] = [
                    cats_master[i] 
                    for i in np.argsort(cats_ids_master) 
                    if cats_master[i]['name'] in filter_settings['category_names']
                ]
            
            # no categories are provided, so assume use ALL available.
            else:

                self.dataset['categories'] = [
                    cats_master[i] 
                    for i in np.argsort(cats_ids_master) 
                ]

                filter_settings['category_names'] = [cat['name'] for cat in self.dataset['categories']]

                trainable_cats = trainable_cats | set(filter_settings['category_names'])
            
            valid_anns = []
            im_height_map = {}

            for im_obj in self.dataset['images']:
                im_height_map[im_obj['id']] = im_obj['height']

            # Filter out annotations
            for anno_idx, anno in enumerate(self.dataset['annotations']):
                
                im_height = im_height_map[anno['image_id']]

                ignore = is_ignore(anno, filter_settings, im_height)
                
                if filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
                    bbox2D =  BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

                elif anno['bbox2D_proj'][0] != -1:
                    bbox2D = BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

                elif anno['bbox2D_tight'][0] != -1:
                    bbox2D = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

                else: 
                    continue

                width = bbox2D[2]
                height = bbox2D[3]

                self.dataset['annotations'][anno_idx]['area'] = width*height
                self.dataset['annotations'][anno_idx]['iscrowd'] = False
                self.dataset['annotations'][anno_idx]['ignore'] = ignore
                self.dataset['annotations'][anno_idx]['ignore2D'] = ignore
                self.dataset['annotations'][anno_idx]['ignore3D'] = ignore
                
                if filter_settings['modal_2D_boxes'] and anno['bbox2D_tight'][0] != -1:
                    self.dataset['annotations'][anno_idx]['bbox'] = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                
                else:
                    self.dataset['annotations'][anno_idx]['bbox'] = bbox2D
                
                self.dataset['annotations'][anno_idx]['bbox3D'] = anno['bbox3D_cam']
                self.dataset['annotations'][anno_idx]['depth'] = anno['center_cam'][2]

                category_name = anno["category_name"]

                # category is part of trainable categories?
                if category_name in trainable_cats:
                    valid_anns.append(self.dataset['annotations'][anno_idx])

            self.dataset['annotations'] = valid_anns

        self.createIndex()

    def info(self):
        
        infos = self.dataset['info']
        if type(infos) == dict:
            infos = [infos]

        for i, info in enumerate(infos):
            print('Dataset {}/{}'.format(i+1, infos))

            for key, value in info.items():
                print('{}: {}'.format(key, value))


def register_and_store_model_metadata(datasets, output_dir, filter_settings=None):

    output_file = os.path.join(output_dir, 'category_meta.json')

    if os.path.exists(output_file):
        metadata = util.load_json(output_file)
        thing_classes = metadata['thing_classes']
        id_map = metadata['thing_dataset_id_to_contiguous_id']

        # json saves id map as strings rather than ints
        id_map = {int(idA):idB for idA, idB in id_map.items()}

    else:
        omni3d_stats = util.load_json(os.path.join('datasets', 'Omni3D', 'stats.json'))
        thing_classes = filter_settings['category_names']

        cat_ids = []
        for cat in thing_classes:
            cat_idx = omni3d_stats['category_names'].index(cat)
            cat_id = omni3d_stats['categories'][cat_idx]['id']
            cat_ids.append(cat_id)

        cat_order = np.argsort(cat_ids)
        cat_ids = [cat_ids[i] for i in cat_order]
        thing_classes = [thing_classes[i] for i in cat_order]
        id_map = {id: i for i, id in enumerate(cat_ids)}
        
        util.save_json(output_file, {
            'thing_classes': thing_classes,
            'thing_dataset_id_to_contiguous_id': id_map,
        })

    MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
    MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id  = id_map


def load_omni3d_json(json_file, image_root, dataset_name, filter_settings, filter_empty=False):
    
    # read in the dataset
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # the global meta information for the full dataset
    meta_model = MetadataCatalog.get('omni3d_model')

    # load the meta information
    meta = MetadataCatalog.get(dataset_name)
    cat_ids = sorted(coco_api.getCatIds(filter_settings['category_names']))
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes
    
    # the id mapping must be based on the model!
    id_map = meta_model.thing_dataset_id_to_contiguous_id
    meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.info(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in Omni3D format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    # annotation keys to pass along
    ann_keys = [
        "bbox", "bbox3D_cam", "bbox2D_proj", "bbox2D_trunc", "bbox2D_tight", 
        "center_cam", "dimensions", "pose", "R_cam", "category_id",
    ]
    
    # optional per image keys to pass if exists
    # this property is unique to KITTI. 
    img_keys_optional = ['p2']

    invalid_count = 0
    
    for (img_dict, anno_dict_list) in imgs_anns:
        
        has_valid_annotation = False

        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_path"])
        record["dataset_id"] = img_dict["dataset_id"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["K"] = img_dict["K"]

        # store optional keys when available
        for img_key in img_keys_optional:
            if img_key in img_dict:
                record[img_key] = img_dict[img_key]

        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            annotation_category_id = obj["category_id"]

            # category is not part of ids and is not in the ignore category?
            if not (annotation_category_id in id_map) and not (anno['category_name'] in filter_settings['ignore_names']):
                continue

            ignore = is_ignore(anno, filter_settings, img_dict["height"])
            
            obj['iscrowd'] = False
            obj['ignore'] = ignore
            
            if filter_settings['modal_2D_boxes'] and 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
                obj['bbox'] = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

            elif filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
                obj['bbox'] =  BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

            elif 'bbox2D_proj' in anno:
                obj['bbox'] = BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

            else:
                continue

            obj['pose'] = anno['R_cam']

            # store category as -1 for ignores!
            obj["category_id"] = -1 if ignore else id_map[annotation_category_id]

            objs.append(obj)

            has_valid_annotation |= (not ignore)

        if has_valid_annotation or (not filter_empty):
            record["annotations"] = objs
            dataset_dicts.append(record)
            
        else:
            invalid_count += 1 
    
    logger.info("Filtered out {}/{} images without valid annotations".format(invalid_count, len(imgs_anns)))

    return dataset_dicts