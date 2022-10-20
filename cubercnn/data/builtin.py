# Copyright (c) Meta Platforms, Inc. and affiliates

def get_omni3d_categories(dataset="omni3d"):
    """
    Returns the Omni3D categories for dataset
    Args:
        dataset: str
    Returns:
        cats: set of strings with category names
    """

    if dataset == "omni3d":
        cats = set({'chair', 'table', 'cabinet', 'car', 'lamp', 'books', 'sofa', 'pedestrian', 'picture', 'window', 'pillow', 'truck', 'door', 'blinds', 'sink', 'shelves', 'television', 'shoes', 'cup', 'bottle', 'bookcase', 'laptop', 'desk', 'cereal box', 'floor mat', 'traffic cone', 'mirror', 'barrier', 'counter', 'camera', 'bicycle', 'toilet', 'bus', 'bed', 'refrigerator', 'trailer', 'box', 'oven', 'clothes', 'van', 'towel', 'motorcycle', 'night stand', 'stove', 'machine', 'stationery', 'bathtub', 'cyclist', 'curtain', 'bin'})
        assert len(cats) == 50
    elif dataset == "omni3d_in":
        cats = set({'stationery', 'sink', 'table', 'floor mat', 'bottle', 'bookcase', 'bin', 'blinds', 'pillow', 'bicycle', 'refrigerator', 'night stand', 'chair', 'sofa', 'books', 'oven', 'towel', 'cabinet', 'window', 'curtain', 'bathtub', 'laptop', 'desk', 'television', 'clothes', 'stove', 'cup', 'shelves', 'box', 'shoes', 'mirror', 'door', 'picture', 'lamp', 'machine', 'counter', 'bed', 'toilet'})
        assert len(cats) == 38
    elif dataset == "omni3d_out":
        cats = set({'cyclist', 'pedestrian', 'trailer', 'bus', 'motorcycle', 'car', 'barrier', 'truck', 'van', 'traffic cone', 'bicycle'})
        assert len(cats) == 11
    elif dataset in ["SUNRGBD_train", "SUNRGBD_val", "SUNRGBD_test"]:
        cats = set({'bicycle', 'books', 'bottle', 'chair', 'cup', 'laptop', 'shoes', 'towel', 'blinds', 'window', 'lamp', 'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'toilet', 'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed', 'night stand', 'pillow', 'sofa', 'television', 'floor mat', 'curtain', 'clothes', 'stationery', 'refrigerator', 'bin', 'stove', 'oven', 'machine'})
        assert len(cats) == 38
    elif dataset in ["Hypersim_train", "Hypersim_val"]:
        cats = set({'books', 'chair', 'towel', 'blinds', 'window', 'lamp', 'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'toilet', 'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed', 'night stand', 'pillow', 'sofa', 'television', 'floor mat', 'curtain', 'clothes', 'stationery', 'refrigerator'})
        assert len(cats) == 29
    elif dataset == "Hypersim_test":
        # Hypersim test annotation does not contain toilet
        cats = set({'books', 'chair', 'towel', 'blinds', 'window', 'lamp', 'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed', 'night stand', 'pillow', 'sofa', 'television', 'floor mat', 'curtain', 'clothes', 'stationery', 'refrigerator'})
        assert len(cats) == 28
    elif dataset in ["ARKitScenes_train", "ARKitScenes_val", "ARKitScenes_test"]:
        cats = set({'table', 'bed', 'sofa', 'television', 'refrigerator', 'chair', 'oven', 'machine', 'stove', 'shelves', 'sink', 'cabinet', 'bathtub', 'toilet'})
        assert len(cats) == 14
    elif dataset in ["Objectron_train", "Objectron_val", "Objectron_test"]:
        cats = set({'bicycle', 'books', 'bottle', 'camera', 'cereal box', 'chair', 'cup', 'laptop', 'shoes'})
        assert len(cats) == 9
    elif dataset in ["KITTI_train", "KITTI_val", "KITTI_test"]:
        cats = set({'pedestrian', 'car', 'cyclist', 'van', 'truck'})
        assert len(cats) == 5
    elif dataset in ["nuScenes_train", "nuScenes_val", "nuScenes_test"]:
        cats = set({'pedestrian', 'car', 'truck', 'traffic cone', 'barrier', 'motorcycle', 'bicycle', 'bus', 'trailer'})
        assert len(cats) == 9
    else:
        raise ValueError("%s dataset is not registered." % (dataset))

    return cats