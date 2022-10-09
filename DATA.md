- [Data Preparation](#data-preparation)  
    - [Download Omni3D json](#download-omni3d-json)
    - [Download Individual Datasets](#download-individual-datasets)
- [Data Usage](#data-usage)  
    - [Coordinate System](#coordinate-system)
    - [Annotation Format](#annotation-format)
    - [Example Loading Data](#example-loading-data)

# Data Preparation

The Omni3D dataset is comprised of 6 datasets which have been pre-processed into the same annotation format and camera coordinate systems. To use a subset or the full dataset you must download:

1. The processed Omni3D json files
2. RGB images from each dataset separately

## Download Omni3D json

Run

```
sh datasets/Omni3D/download_omni3d_json.sh
```

to download and extract the Omni3D train, val and test json annotation files.

## Download Individual Datasets

Below are the instructions for setting up each individual dataset. It is recommended to download only the data you plan to use.  

### KITTI
Download the left color images from [KITTI's official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Unzip or softlink the images into the root `./Omni3D/` which should have the folder structure as detailed below. Note that we only require the image_2 folder. 

```bash
datasets/KITTI_object
└── training
    ├── image_2
```


### nuScenes

Download the trainval images from the [official nuScenes website](https://www.nuscenes.org/nuscenes#download). Unzip or softlink the images into the root `./Omni3D/` which should have the folder structure as detailed below. Note that we only require the CAM_FRONT folder.

```bash
datasets/nuScenes/samples
└── samples
    ├── CAM_FRONT
```

### Objectron

Run

```
sh datasets/objectron/download_objectron_images.sh
```

to download and extract the Objectron pre-processed images (~24 GB).

### SUN RGB-D

Download the "SUNRGBD V1" images at [SUN RGB-D's official website](https://rgbd.cs.princeton.edu/). Unzip or softlink the images into the root `./Omni3D/` which should have the folder structure as detailed below. 

```bash
./Omni3D/datasets/SUNRGBD
├── kv1
├── kv2
├── realsense
```

### ARKitScenes

Run

```
sh datasets/ARKitScenes/download_arkitscenes_images.sh
```

to download and extract the ARKitScenes pre-processed images (~28 GB).

### Hypersim

Follow the [download instructions](https://github.com/apple/ml-hypersim/tree/main/contrib/99991) from [Thomas Germer](https://github.com/99991) in order to download all \*tonemap.jpg preview images in order to avoid downloading the full Hypersim dataset. For example:

```bash
git clone https://github.com/apple/ml-hypersim
cd ml-hypersim/
python contrib/99991/download.py -c .tonemap.jpg -d /path/to/Omni3D/datasets/hypersim --silent
```

Then arrange or unzip the downloaded images into the root `./Omni3D/` so that it has the below folder structure.

```bash
datasets/hypersim/
├── ai_001_001
├── ai_001_002
├── ai_001_003
├── ai_001_004
├── ai_001_005
├── ai_001_006
...
```

# Data Usage

Below we describe the unified 3D annotation coordinate systems, annotation format, and an example script. 


## Coordinate System

All 3D annotations are provided in a shared camera coordinate system with 
+x right, +y down, +z toward screen. 

The vertex order of bbox3D_cam:
```
                v4_____________________v5
                /|                    /|
               / |                   / |
              /  |                  /  |
             /___|_________________/   |
          v0|    |                 |v1 |
            |    |                 |   |
            |    |                 |   |
            |    |                 |   |
            |    |_________________|___|
            |   / v7               |   /v6
            |  /                   |  /
            | /                    | /
            |/_____________________|/
            v3                     v2
```

## Annotation Format
Each dataset is formatted as a dict in python in the below format.

```python
dataset {
    "info"			: info,
    "images"			: [image],
    "categories"		: [category],
    "annotations"		: [object],
}

info {
	"id"			: str,
	"source"		: int,
	"name"			: str,
	"split"			: str,
	"version"		: str,
	"url"			: str,
}

image {
	"id"			: int,
	"dataset_id"		: int,
	"width"			: int,
	"height"		: int,
	"file_path"		: str,
	"K"			: list (3x3),
	"src_90_rotate"		: int,					# im was rotated X times, 90 deg counterclockwise 
	"src_flagged"		: bool,					# flagged as potentially inconsistent sky direction
}

category {
	"id"			: int,
	"name"			: str,
	"supercategory"	: str
}

object {
	
	"id"			: int,					# unique annotation identifier
	"image_id"		: int,					# identifier for image
	"category_id"		: int,					# identifier for the category
	"category_name"		: str,					# plain name for the category
	
	# General 2D/3D Box Parameters.
	# Values are set to -1 when unavailable.
	"valid3D"		: bool,				        # flag for no reliable 3D box
	"bbox2D_tight"		: [x1, y1, x2, y2],			# 2D corners of annotated tight box
	"bbox2D_proj"		: [x1, y1, x2, y2],			# 2D corners projected from bbox3D
	"bbox2D_trunc"		: [x1, y1, x2, y2],			# 2D corners projected from bbox3D then truncated
	"bbox3D_cam"		: [[x1, y1, z1]...[x8, y8, z8]]		# 3D corners in meters and camera coordinates
	"center_cam"		: [x, y, z],				# 3D center in meters and camera coordinates
	"dimensions"		: [width, height, length],		# 3D attributes for object dimensions in meters
	"R_cam"			: list (3x3),				# 3D rotation matrix to the camera frame rotation
	
	# Optional dataset specific properties,
	# used mainly for evaluation and ignore.
	# Values are set to -1 when unavailable.
	"behind_camera"		: bool,					# a corner is behind camera
	"visibility"		: float, 				# annotated visibility 0 to 1
	"truncation"		: float, 				# computed truncation 0 to 1
	"segmentation_pts"	: int, 					# visible instance segmentation points
	"lidar_pts" 		: int, 					# visible LiDAR points in the object
	"depth_error"		: float,				# L1 of depth map and rendered object
}
```


## Example Loading Data
Each dataset is named as "Omni3D_{name}_{split}.json" where split can be train, val, or test. 

The annotations are in a COCO-like format such that if you load the json from the Omni3D class which inherits the [COCO class](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L70), you can use basic COCO dataset functions as demonstrated with the below code. 

```python
from cubercnn import data

dataset_paths_to_json = ['path/to/Omni3D/{name}_{split}.json', ...]

# Example 1. load all images
dataset = data.Omni3D(dataset_paths_to_json)
imgIds = dataset.getImgIds()
imgs = dataset.loadImgs(imgIds)

# Example 2. load annotations for image index 0
annIds = dataset.getAnnIds(imgIds=imgs[0]['id'])
anns = dataset.loadAnns(annIds)
```