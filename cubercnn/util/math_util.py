# Copyright (c) Meta Platforms, Inc. and affiliates
import math
import numpy as np
import pandas as pd
from typing import Tuple, List
from copy import copy
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import SoftPhongShader
import cv2
import torch 
from pytorch3d.structures import Meshes
from detectron2.structures import BoxMode
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures.meshes import (
    Meshes,
)

from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings,
    MeshRasterizer
)

from pytorch3d.renderer import (
    PerspectiveCameras, 
    SoftSilhouetteShader, 
    RasterizationSettings,
    MeshRasterizer
)
from detectron2.data import (
    MetadataCatalog,
)
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import MeshRenderer as MR

UNIT_CUBE = np.array([
       [-0.5, -0.5, -0.5],
       [ 0.5, -0.5, -0.5],
       [ 0.5,  0.5, -0.5],
       [-0.5,  0.5, -0.5],
       [-0.5, -0.5,  0.5],
       [ 0.5, -0.5,  0.5],
       [ 0.5,  0.5,  0.5],
       [-0.5,  0.5,  0.5]
])

def upto_2Pi(val):

    out = val

    # constrain between [0, 2pi)
    while out >= 2*math.pi: out -= math.pi * 2
    while out < 0: out += math.pi * 2

    return out

def upto_Pi(val):

    out = val

    # constrain between [0, pi)
    while out >= math.pi: out -= math.pi
    while out < 0: out += math.pi

    return out

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def mat2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    #singular = sy < 1e-6

    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], sy)
    z = math.atan2(R[1, 0], R[0, 0])

    return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles.
# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def euler2mat(euler):

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(euler[0]), -math.sin(euler[0])],
                    [0, math.sin(euler[0]), math.cos(euler[0])]
                    ])

    R_y = np.array([[math.cos(euler[1]), 0, math.sin(euler[1])],
                    [0, 1, 0],
                    [-math.sin(euler[1]), 0, math.cos(euler[1])]
                    ])

    R_z = np.array([[math.cos(euler[2]), -math.sin(euler[2]), 0],
                    [math.sin(euler[2]), math.cos(euler[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()

def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces from a 3D cuboid representation.
    Args:
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
    Returns:
        verts: the 3D vertices of the cuboid in camera space
        faces: the vertex indices per face
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1]

    # make sure types are correct
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)

    x3d = box3d[:, 0].unsqueeze(1)
    y3d = box3d[:, 1].unsqueeze(1)
    z3d = box3d[:, 2].unsqueeze(1)
    w3d = box3d[:, 3].unsqueeze(1)
    h3d = box3d[:, 4].unsqueeze(1)
    l3d = box3d[:, 5].unsqueeze(1)

    '''
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
    '''

    verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device))

    # setup X
    verts[:, 0, [0, 3, 4, 7]] = -l3d / 2
    verts[:, 0, [1, 2, 5, 6]] = l3d / 2

    # setup Y
    verts[:, 1, [0, 1, 4, 5]] = -h3d / 2
    verts[:, 1, [2, 3, 6, 7]] = h3d / 2

    # setup Z
    verts[:, 2, [0, 1, 2, 3]] = -w3d / 2
    verts[:, 2, [4, 5, 6, 7]] = w3d / 2

    if R is not None:

        # rotate
        verts = R @ verts
    
    # translate
    verts[:, 0, :] += x3d
    verts[:, 1, :] += y3d
    verts[:, 2, :] += z3d

    verts = verts.transpose(1, 2)

    faces = torch.tensor([
        [0, 1, 2], # front TR
        [2, 3, 0], # front BL

        [1, 5, 6], # right TR
        [6, 2, 1], # right BL

        [4, 0, 3], # left TR
        [3, 7, 4], # left BL

        [5, 4, 7], # back TR
        [7, 6, 5], # back BL

        [4, 5, 1], # top TR
        [1, 0, 4], # top BL

        [3, 2, 6], # bottom TR
        [6, 7, 3], # bottom BL
    ]).float().unsqueeze(0).repeat([n, 1, 1])

    if squeeze:
        verts = verts.squeeze()
        faces = faces.squeeze()

    return verts, faces.to(verts.device)

def get_cuboid_verts(K, box3d, R=None, view_R=None, view_T=None):

    # make sure types are correct
    K = to_float_tensor(K)
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)

    n = len(box3d)

    if len(K.shape) == 2:
        K = K.unsqueeze(0).repeat([n, 1, 1])

    corners_3d, _ = get_cuboid_verts_faces(box3d, R)
    if view_T is not None:
        corners_3d -= view_T.view(1, 1, 3)
    if view_R is not None:
        corners_3d = (view_R @ corners_3d[0].T).T.unsqueeze(0)
    if view_T is not None:
        corners_3d[:, :, -1] += view_T.view(1, 1, 3)[:, :, -1]*1.25

    # project to 2D
    corners_2d = K @ corners_3d.transpose(1, 2)
    corners_2d[:, :2, :] = corners_2d[:, :2, :] / corners_2d[:, 2, :].unsqueeze(1)
    corners_2d = corners_2d.transpose(1, 2)

    if squeeze:
        corners_3d = corners_3d.squeeze()
        corners_2d = corners_2d.squeeze()

    return corners_2d, corners_3d


def approx_eval_resolution(h, w, scale_min=0, scale_max=1e10):
    """
    Approximates the resolution an image with h x w resolution would
    run through a model at which constrains the scale to a min and max. 
    Args:
        h (int): input resolution height
        w (int): input resolution width
        scale_min (int): minimum scale allowed to resize too
        scale_max (int): maximum scale allowed to resize too
    Returns:
        h (int): output resolution height
        w (int): output resolution width
        sf (float): scaling factor that was applied
            which can convert from original --> network resolution.
    """
    orig_h = h

    # first resize to min
    sf = scale_min / min(h, w)
    h *= sf
    w *= sf

    # next resize to max
    sf = min(scale_max / max(h, w), 1.0)
    h *= sf
    w *= sf

    return h, w, h/orig_h


def compute_priors(cfg, datasets, max_cluster_rounds=1000, min_points_for_std=5):
    """
    Computes priors via simple averaging or a custom K-Means clustering. 
    """

    annIds = datasets.getAnnIds()
    anns = datasets.loadAnns(annIds)

    data_raw = []

    category_names = MetadataCatalog.get('omni3d_model').thing_classes

    virtual_depth = cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH
    virtual_focal = cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL
    test_scale_min = cfg.INPUT.MIN_SIZE_TEST
    test_scale_max = cfg.INPUT.MAX_SIZE_TEST

    '''
    Accumulate the annotations while discarding the 2D center information
    (hence, keeping only the 2D and 3D scale information, and properties.)
    '''

    for ann_idx, ann in enumerate(anns):

        category_name = ann['category_name'].lower()

        ignore = ann['ignore']
        dataset_id = ann['dataset_id']
        image_id = ann['image_id']

        fy = datasets.imgs[image_id]['K'][1][1]
        im_h = datasets.imgs[image_id]['height']
        im_w = datasets.imgs[image_id]['width']
        f = 2 * fy / im_h

        if cfg.DATASETS.MODAL_2D_BOXES and 'bbox2D_tight' in ann and ann['bbox2D_tight'][0] != -1:
            x, y, w, h =  BoxMode.convert(ann['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

        elif cfg.DATASETS.TRUNC_2D_BOXES and 'bbox2D_trunc' in ann and not np.all([val==-1 for val in ann['bbox2D_trunc']]):
            x, y, w, h =  BoxMode.convert(ann['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

        elif 'bbox2D_proj' in ann:
            x, y, w, h =  BoxMode.convert(ann['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

        else:
            continue

        x3d, y3d, z3d = ann['center_cam']
        w3d, h3d, l3d = ann['dimensions']
        
        test_h, test_w, sf = approx_eval_resolution(im_h, im_w, test_scale_min, test_scale_max)

        # scale everything to test resolution
        h *= sf
        w *= sf

        if virtual_depth:
            virtual_to_real = compute_virtual_scale_from_focal_spaces(fy, im_h, virtual_focal, test_h)
            real_to_virtual = 1/virtual_to_real
            z3d *= real_to_virtual

        scale = np.sqrt(h**2 + w**2)

        if (not ignore) and category_name in category_names:
            data_raw.append([category_name, w, h, x3d, y3d, z3d, w3d, h3d, l3d, w3d*h3d*l3d, dataset_id, image_id, fy, f, scale])

    # TODO pandas is fairly inefficient to rely on for large scale.
    df_raw = pd.DataFrame(data_raw, columns=[
        'name', 
        'w', 'h', 'x3d', 'y3d', 'z3d', 
        'w3d', 'h3d', 'l3d', 'volume', 
        'dataset', 'image', 
        'fy', 'f', 'scale'
    ])

    priors_bins = []
    priors_dims_per_cat = []
    priors_z3d_per_cat = []
    priors_y3d_per_cat = []

    # compute priors for z and y globally
    priors_z3d = [df_raw.z3d.mean(), df_raw.z3d.std()]
    priors_y3d = [df_raw.y3d.mean(), df_raw.y3d.std()]

    n_bins = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS

    # Each prior is pre-computed per category
    for cat in category_names:
        
        df_cat = df_raw[df_raw.name == cat]        

        '''
        First compute static variable statistics
        '''

        scales = torch.FloatTensor(np.array(df_cat.scale))
        n = len(scales)

        if n > 0:
            priors_dims_per_cat.append([[df_cat.w3d.mean(), df_cat.h3d.mean(), df_cat.l3d.mean()], [df_cat.w3d.std(), df_cat.h3d.std(), df_cat.l3d.std()]])            
            priors_z3d_per_cat.append([df_cat.z3d.mean(), df_cat.z3d.std()])            
            priors_y3d_per_cat.append([df_cat.y3d.mean(), df_cat.y3d.std()])
        
        else:
            # dummy data.
            priors_dims_per_cat.append([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])            
            priors_z3d_per_cat.append([50, 50])            
            priors_y3d_per_cat.append([1, 10])

        '''
        Next compute Z cluster statistics based on y and area
        '''

        def compute_cluster_scale_mean(scales, assignments, n_bins, match_quality):

            cluster_scales = []

            for bin in range(n_bins):

                in_cluster = assignments==bin
                
                if in_cluster.sum() < min_points_for_std:
                    in_cluster[match_quality[:, bin].topk(min_points_for_std)[1]] = True

                scale = scales[in_cluster].mean()
                cluster_scales.append(scale.item())

            return torch.FloatTensor(cluster_scales)

        if n_bins > 1:

            if n < min_points_for_std:
                
                print('Warning {} category has only {} valid samples...'.format(cat, n))
                
                # dummy data since category doesn't have available samples.
                max_scale = cfg.MODEL.ANCHOR_GENERATOR.SIZES[-1][-1]
                min_scale = cfg.MODEL.ANCHOR_GENERATOR.SIZES[0][0]
                base = (max_scale / min_scale) ** (1 / (n_bins - 1))
                cluster_scales = np.array([min_scale * (base ** i) for i in range(0, n_bins)])
                
                # default values are unused anyways in training. but range linearly 
                # from 100 to 1 and ascend with 2D scale. 
                bin_priors_z = [[b, 15] for b in np.arange(100, 1, -(100-1)/n_bins)]
                priors_bins.append((cat, cluster_scales.tolist(), bin_priors_z))
                assert len(bin_priors_z) == n_bins, 'Broken default bin scaling.'
            else:
            
                max_scale = scales.max()
                min_scale = scales.min()
                base = (max_scale / min_scale) ** (1 / (n_bins - 1))
                cluster_scales = torch.FloatTensor([min_scale * (base ** i) for i in range(0, n_bins)])

                best_score = -np.inf

                for round in range(max_cluster_rounds):
                    
                    # quality scores for gts and clusters (n x n_bins)
                    match_quality = -(cluster_scales.unsqueeze(0) - scales.unsqueeze(1)).abs()

                    # assign to best clusters
                    scores, assignments_round = match_quality.max(1)
                    round_score = scores.mean().item()

                    if np.round(round_score, 5) > best_score:
                        best_score = round_score
                        assignments = assignments_round
                        
                        # make new clusters
                        cluster_scales = compute_cluster_scale_mean(scales, assignments, n_bins, match_quality)

                    else:
                        break

                bin_priors_z = []

                for bin in range(n_bins):
                    
                    in_cluster = assignments == bin

                    # not enough in the cluster to compute reliable stats?
                    # fill it with the topk others
                    if in_cluster.sum() < min_points_for_std:
                        in_cluster[match_quality[:, bin].topk(min_points_for_std)[1]] = True

                    # move to numpy for indexing pandas
                    in_cluster = in_cluster.numpy()

                    z3d_mean = df_cat.z3d[in_cluster].mean()
                    z3d_std = df_cat.z3d[in_cluster].std()

                    bin_priors_z.append([z3d_mean, z3d_std])
                
                priors_bins.append((cat, cluster_scales.numpy().tolist(), bin_priors_z))
        
    priors = {
        'priors_dims_per_cat': priors_dims_per_cat,
        'priors_z3d_per_cat': priors_z3d_per_cat,
        'priors_y3d_per_cat': priors_y3d_per_cat,
        'priors_bins': priors_bins,
        'priors_y3d': priors_y3d,
        'priors_z3d': priors_z3d,
    }
    
    return priors

def convert_3d_box_to_2d(K, box3d, R=None, clipw=0, cliph=0, XYWH=True, min_z=0.20):
    """
    Converts a 3D box to a 2D box via projection. 
    Args:
        K (np.array): intrinsics matrix 3x3
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
        clipw (int): clip invalid X to the image bounds. Image width is usually used here.
        cliph (int): clip invalid Y to the image bounds. Image height is usually used here.
        XYWH (bool): returns in XYWH if true, otherwise XYXY format. 
        min_z: the threshold for how close a vertex is allowed to be before being
            considered as invalid for projection purposes.
    Returns:
        box2d (flexible): the 2D box results.
        behind_camera (bool): whether the projection has any points behind the camera plane.
        fully_behind (bool): all points are behind the camera plane. 
    """

    # bounds used for vertices behind image plane
    topL_bound = torch.tensor([[0, 0, 0]]).float()
    topR_bound = torch.tensor([[clipw-1, 0, 0]]).float()
    botL_bound = torch.tensor([[0, cliph-1, 0]]).float()
    botR_bound = torch.tensor([[clipw-1, cliph-1, 0]]).float()

    # make sure types are correct
    K = to_float_tensor(K)
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)
    verts2d, verts3d = get_cuboid_verts(K, box3d, R)

    # any boxes behind camera plane?
    verts_behind = verts2d[:, :, 2] <= min_z
    behind_camera = verts_behind.any(1)

    verts_signs = torch.sign(verts3d)

    # check for any boxes projected behind image plane corners
    topL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] < 0)
    topR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] < 0)
    botL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] > 0)
    botR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] > 0)
    
    # clip values to be in bounds for invalid points
    verts2d[topL] = topL_bound
    verts2d[topR] = topR_bound
    verts2d[botL] = botL_bound
    verts2d[botR] = botR_bound

    x, xi = verts2d[:, :, 0].min(1)
    y, yi = verts2d[:, :, 1].min(1)
    x2, x2i = verts2d[:, :, 0].max(1)
    y2, y2i = verts2d[:, :, 1].max(1)

    fully_behind = verts_behind.all(1)

    width = x2 - x
    height = y2 - y

    if XYWH:
        box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1)), dim=1)
    else:
        box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)

    if squeeze:
        box2d = box2d.squeeze()
        behind_camera = behind_camera.squeeze()
        fully_behind = fully_behind.squeeze()

    return box2d, behind_camera, fully_behind


# 
def compute_virtual_scale_from_focal_spaces(f, H, f0, H0):
    """
    Computes the scaling factor of depth from f0, H0 to f, H
    Args:
        f (float): the desired [virtual] focal length (px)
        H (float): the desired [virtual] height (px)
        f0 (float): the initial [real] focal length (px)
        H0 (float): the initial [real] height (px)
    Returns:
        the scaling factor float to convert form (f0, H0) --> (f, H)
    """
    return (H0 * f) / (f0 * H)


def R_to_allocentric(K, R, u=None, v=None):
    """
    Convert a rotation matrix or series of rotation matrices to allocentric
    representation given a 2D location (u, v) in pixels. 
    When u or v are not available, we fall back on the principal point of K.
    """
    if type(K) == torch.Tensor:
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        sx = K[:, 0, 2]
        sy = K[:, 1, 2]

        n = len(K)
        
        oray = torch.stack(((u - sx)/fx, (v - sy)/fy, torch.ones_like(u))).T
        oray = oray / torch.linalg.norm(oray, dim=1).unsqueeze(1)
        angle = torch.acos(oray[:, -1])

        axis = torch.zeros_like(oray)
        axis[:, 0] = axis[:, 0] - oray[:, 1]
        axis[:, 1] = axis[:, 1] + oray[:, 0]
        norms = torch.linalg.norm(axis, dim=1)

        valid_angle = angle > 0

        M = axis_angle_to_matrix(angle.unsqueeze(1)*axis/norms.unsqueeze(1))
        
        R_view = R.clone()
        R_view[valid_angle] = torch.bmm(M[valid_angle].transpose(2, 1), R[valid_angle])

    else:
        fx = K[0][0]
        fy = K[1][1]
        sx = K[0][2]
        sy = K[1][2]
        
        if u is None:
            u = sx

        if v is None:
            v = sy

        oray = np.array([(u - sx)/fx, (v - sy)/fy, 1])
        oray = oray / np.linalg.norm(oray)
        cray = np.array([0, 0, 1])
        angle = math.acos(cray.dot(oray))
        if angle != 0:
            axis = np.cross(cray, oray)
            axis_torch = torch.from_numpy(angle*axis/np.linalg.norm(axis)).float()
            R_view = np.dot(axis_angle_to_matrix(axis_torch).numpy().T, R)
        else: 
            R_view = R

    return R_view


def R_from_allocentric(K, R_view, u=None, v=None):
    """
    Convert a rotation matrix or series of rotation matrices to egocentric
    representation given a 2D location (u, v) in pixels. 
    When u or v are not available, we fall back on the principal point of K.
    """
    if type(K) == torch.Tensor:
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        sx = K[:, 0, 2]
        sy = K[:, 1, 2]

        n = len(K)
        
        oray = torch.stack(((u - sx)/fx, (v - sy)/fy, torch.ones_like(u))).T
        oray = oray / torch.linalg.norm(oray, dim=1).unsqueeze(1)
        angle = torch.acos(oray[:, -1])

        axis = torch.zeros_like(oray)
        axis[:, 0] = axis[:, 0] - oray[:, 1]
        axis[:, 1] = axis[:, 1] + oray[:, 0]
        norms = torch.linalg.norm(axis, dim=1)

        valid_angle = angle > 0

        M = axis_angle_to_matrix(angle.unsqueeze(1)*axis/norms.unsqueeze(1))
        
        R = R_view.clone()
        R[valid_angle] = torch.bmm(M[valid_angle], R_view[valid_angle])

    else:
        fx = K[0][0]
        fy = K[1][1]
        sx = K[0][2]
        sy = K[1][2]
        
        if u is None:
            u = sx

        if v is None:
            v = sy

        oray = np.array([(u - sx)/fx, (v - sy)/fy, 1])
        oray = oray / np.linalg.norm(oray)
        cray = np.array([0, 0, 1])
        angle = math.acos(cray.dot(oray))
        if angle != 0:
            #axis = np.cross(cray, oray)
            axis = np.array([-oray[1], oray[0], 0])
            axis_torch = torch.from_numpy(angle*axis/np.linalg.norm(axis)).float()
            R = np.dot(axis_angle_to_matrix(axis_torch).numpy(), R_view)
        else: 
            R = R_view

    return R

def render_depth_map(K, box3d, pose, width, height, device=None):
    
    cameras = get_camera(K, width, height)
    renderer = get_basic_renderer(cameras, width, height)

    mesh = mesh_cuboid(box3d, pose)

    if device is not None:
        cameras = cameras.to(device)
        renderer = renderer.to(device)
        mesh = mesh.to(device)

    im_rendered, fragment = renderer(mesh)
    silhouettes = im_rendered[:, :, :, -1] > 0

    zbuf = fragment.zbuf[:, :, :, 0]
    zbuf[zbuf==-1] = math.inf
    depth_map, depth_map_inds = zbuf.min(dim=0)

    return silhouettes, depth_map, depth_map_inds

def estimate_visibility(K, box3d, pose, width, height, device=None):

    silhouettes, depth_map, depth_map_inds = render_depth_map(K, box3d, pose, width, height, device=device)

    n = silhouettes.shape[0]

    visibilies = []

    for annidx in range(n):

        area = silhouettes[annidx].sum()
        visible = (depth_map_inds[silhouettes[annidx]] == annidx).sum()

        visibilies.append((visible / area).item())

    return visibilies

def estimate_truncation(K, box3d, R, imW, imH):

    box2d, out_of_bounds, fully_behind =  convert_3d_box_to_2d(K, box3d, R, imW, imH)
    
    if fully_behind:
        return 1.0

    box2d = box2d.detach().cpu().numpy().tolist()
    box2d_XYXY = BoxMode.convert(box2d, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    image_box = np.array([0, 0, imW-1, imH-1])

    truncation = 1 - iou(np.array(box2d_XYXY)[np.newaxis], image_box[np.newaxis], ign_area_b=True)

    return truncation.item()


def mesh_cuboid(box3d=None, R=None, color=None):

    verts, faces = get_cuboid_verts_faces(box3d, R)
    
    if verts.ndim == 2:
        verts = to_float_tensor(verts).unsqueeze(0)
        faces = to_float_tensor(faces).unsqueeze(0)

    ninstances = len(verts)

    if (isinstance(color, Tuple) or isinstance(color, List)) and len(color) == 3:
        color = torch.tensor(color).view(1, 1, 3).expand(ninstances, 8, 3).float()

    # pass in a tensor of colors per box
    elif color.ndim == 2: 
        color = to_float_tensor(color).unsqueeze(1).expand(ninstances, 8, 3).float()

    device = verts.device

    mesh = Meshes(verts=verts, faces=faces, textures=None if color is None else TexturesVertex(verts_features=color).to(device))

    return mesh

def get_camera(K, width, height, switch_hands=True, R=None, T=None):

    K = to_float_tensor(K)

    if switch_hands:
        K = K @ torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]).float()

    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    if R is None:
        camera = PerspectiveCameras(
            focal_length=((fx, fy),), principal_point=((px, py),), 
            image_size=((height, width),), in_ndc=False
        )
    else:
        camera = PerspectiveCameras(
            focal_length=((fx, fy),), principal_point=((px, py),), 
            image_size=((height, width),), in_ndc=False, R=R, T=T
        )

    return camera


def get_basic_renderer(cameras, width, height, use_color=False):

    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=0 if use_color else np.log(1. / 1e-4 - 1.) * 1e-4, 
        faces_per_pixel=1, 
        perspective_correct=False,
    )

    if use_color:
        # SoftPhongShader, HardPhongShader, HardFlatShader, SoftGouraudShader
        lights = PointLights(location=[[0.0, 0.0, 0.0]])
        shader = SoftPhongShader(cameras=cameras, lights=lights)
    else:
        shader = SoftSilhouetteShader()

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=shader
    )

    return renderer

class MeshRenderer(MR):
    def __init__(self, rasterizer, shader):
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments

def iou(box_a, box_b, mode='cross', ign_area_b=False):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.
    Args:
        box_a (array or tensor): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (array or tensor): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'cross' or 'list', where cross will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        ign_area_b (bool): if true then we ignore area of b. e.g., checking % box a is inside b
    """

    data_type = type(box_a)

    # this mode computes the IoU in the sense of cross.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'cross':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0)
            if not ign_area_b:
                union = union + area_b.unsqueeze(1) - inter

            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) 
            if not ign_area_b:
                union = union + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


def intersect(box_a, box_b, mode='cross'):
    """
    Computes the amount of intersect between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'cross' or 'list', where cross will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    data_type = type(box_a)

    # this mode computes the intersect in the sense of cross.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'cross':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.max(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


def scaled_sigmoid(vals, min=0.0, max=1.0):
    """
    Simple helper function for a scaled sigmoid. 
    The output is bounded by (min, max)
    Args:
        vals (Tensor): input logits to scale
        min (Tensor or float): the minimum value to scale to.
        max (Tensor or float): the maximum value to scale to.
    """
    return min + (max-min)*torch.sigmoid(vals)