# Copyright (c) Meta Platforms, Inc. and affiliates
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import torch
from copy import deepcopy
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.transforms.so3 import (
    so3_relative_angle,
)
from matplotlib.path import Path

from cubercnn import util

def interp_color(dist, bounds=[0, 1], color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    percent = (dist - bounds[0]) / (bounds[1] - bounds[0])
    b = color_lo[0] * (1 - percent) + color_hi[0] * percent
    g = color_lo[1] * (1 - percent) + color_hi[1] * percent
    r = color_lo[2] * (1 - percent) + color_hi[2] * percent

    return (b, g, r)

def draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2):

    w = l3d * scale
    l = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d*-1

    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += w/2 + x + canvas_bev.shape[1] / 2
    corners2[:, 1] += l/2 + z

    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)


def draw_line(im, v0, v1, color=(0, 200, 200), thickness=1):
    cv2.line(im, (int(v0[0]), int(v0[1])), (int(v1[0]), int(v1[1])), color, thickness)


def create_colorbar(height, width, color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    im = np.zeros([height, width, 3])

    for h in range(0, height):

        color = interp_color(h + 0.5, [0, height], color_hi, color_lo)
        im[h, :, 0] = (color[0])
        im[h, :, 1] = (color[1])
        im[h, :, 2] = (color[2])

    return im.astype(np.uint8)


def visualize_from_instances(detections, dataset, dataset_name, min_size_test, output_folder, category_names_official, iteration=''):
    
    vis_folder = os.path.join(output_folder, 'vis')
    
    util.mkdir_if_missing(vis_folder)

    log_str = ''

    xy_errors = []
    z_errors = []
    w3d_errors = []
    h3d_errors = []
    l3d_errors = []
    dim_errors = []
    ry_errors = []

    n_cats = len(category_names_official)
    thres = np.sqrt(1/n_cats)

    for imind, im_obj in enumerate(detections):
        
        write_sample = ((imind % 50) == 0)
        
        annos = dataset._dataset[imind]['annotations']
        gt_boxes_2d = np.array([anno['bbox'] for anno in annos])
        
        if len(gt_boxes_2d)==0: 
            continue

        gt_boxes_2d[:, 2] += gt_boxes_2d[:, 0]
        gt_boxes_2d[:, 3] += gt_boxes_2d[:, 1]

        gt_boxes_cat = np.array([anno['category_id'] for anno in annos])

        if write_sample:
            data_obj = dataset[imind]
            assert(data_obj['image_id'] == im_obj['image_id'])
            im = util.imread(data_obj['file_name'])

        K = np.array(im_obj['K'])
        K_inv = np.linalg.inv(K)

        sf = im_obj['height'] / min_size_test

        for instance in im_obj['instances']:
            
            cat = category_names_official[instance['category_id']]
            score = instance['score']
            x1, y1, w, h = instance['bbox']
            x2 = x1 + w 
            y2 = y1 + h 

            alpha, h3d, w3d, l3d, x3d, y3d, z3d, ry3d = (-1,)*8

            w3d, h3d, l3d = instance['dimensions']

            # unproject
            cen_2d = np.array(instance['center_2D'] + [1])
            z3d = instance['center_cam'][2]

            # get rotation (y-axis only)
            ry3d = np.array(instance['pose'])

            valid_gt_inds = np.flatnonzero(instance['category_id'] == gt_boxes_cat)

            if len(valid_gt_inds) > 0:
                quality_matrix = util.iou(np.array([[x1, y1, x2, y2]]), gt_boxes_2d[valid_gt_inds])
                nearest_gt = quality_matrix.argmax(axis=1)[0]
                nearest_gt_iou = quality_matrix.max(axis=1)[0]
                valid_match = nearest_gt_iou >= 0.5
            else:
                valid_match = False

            if valid_match:
                gt_x1, gt_y1, gt_w, gt_h = annos[valid_gt_inds[nearest_gt]]['bbox']
                gt_x3d, gt_y3d, gt_z3d = annos[valid_gt_inds[nearest_gt]]['center_cam']
                gt_w3d, gt_h3d, gt_l3d = annos[valid_gt_inds[nearest_gt]]['dimensions']
                gt_cen_2d = K @ np.array([gt_x3d, gt_y3d, gt_z3d])
                gt_cen_2d /= gt_cen_2d[2]
                gt_pose = annos[valid_gt_inds[nearest_gt]]['pose']
                gt_ry3d = np.array(gt_pose)

            if valid_match:
            
                # compute errors
                xy_errors.append(np.sqrt(((cen_2d[:2] - gt_cen_2d[:2])**2).sum()))
                z_errors.append(np.abs(z3d - gt_z3d))
                w3d_errors.append(np.abs(w3d - gt_w3d))
                h3d_errors.append(np.abs(h3d - gt_h3d))
                l3d_errors.append(np.abs(l3d - gt_l3d))
                dim_errors.append(np.sqrt((w3d - gt_w3d)**2 + (h3d - gt_h3d)**2 + (l3d - gt_l3d)**2))
                
                try:
                    ry_errors.append(so3_relative_angle(torch.from_numpy(ry3d).unsqueeze(0), torch.from_numpy(gt_ry3d).unsqueeze(0), cos_bound=1).item())
                except:
                    pass

            # unproject point to 3D
            x3d, y3d, z3d = (K_inv @ (z3d*cen_2d))

            # let us visualize the detections now
            if write_sample and score > thres:
                color = util.get_color(instance['category_id'])
                draw_3d_box(im, K, [x3d, y3d, z3d, w3d, h3d, l3d], ry3d, color=color, thickness=int(np.round(3*im.shape[0]/500)), draw_back=False)
                draw_text(im, '{}, z={:.1f}, s={:.2f}'.format(cat, z3d, score), [x1, y1, w, h], scale=0.50*im.shape[0]/500, bg_color=color)

        if write_sample:
            util.imwrite(im, os.path.join(vis_folder, '{:06d}.jpg'.format(imind)))
    
    # safety in case all rotation matrices failed. 
    if len(ry_errors) == 0:
        ry_errors = [1000, 1000]

    log_str += dataset_name + 'iter={}, xy({:.2f}), z({:.2f}), whl({:.2f}, {:.2f}, {:.2f}), ry({:.2f})\n'.format(
        iteration,
        np.mean(xy_errors), np.mean(z_errors),
        np.mean(w3d_errors), np.mean(h3d_errors), np.mean(l3d_errors),
        np.mean(ry_errors),
    )

    return log_str


def imshow(im, fig_num=None):

    if fig_num is not None: plt.figure(fig_num)

    if len(im.shape) == 2:
        im = np.tile(im, [3, 1, 1]).transpose([1, 2, 0])

    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show()


def draw_scene_view(im, K, meshes, text=None, scale=1000, R=None, T=None, zoom_factor=1.0, mode='front_and_novel', blend_weight=0.80, blend_weight_overlay=1.0, ground_bounds=None, canvas=None, zplane=0.05):
    """
    Draws a scene from multiple different modes. 
    Args:
        im (array): the image to draw onto
        K (array): the 3x3 matrix for projection to camera to screen
        meshes ([Mesh]): a list of meshes to draw into the scene
        text ([str]): optional strings to draw per mesh
        scale (int): the size of the square novel view canvas (pixels)
        R (array): a single 3x3 matrix defining the novel view
        T (array): a 3x vector defining the position of the novel view
        zoom_factor (float): an optional amount to zoom out (>1) or in (<1)
        mode (str): supports ['2D_only', 'front', 'novel', 'front_and_novel'] where 
            front implies the front-facing camera view and novel is based on R,T
        blend_weight (float): blend factor for box edges over the RGB
        blend_weight_overlay (float): blends the RGB image with the rendered meshes
        ground_bounds (tuple): max_y3d, x3d_start, x3d_end, z3d_start, z3d_end for the Ground floor or 
            None to let the renderer to estimate the ground bounds in the novel view itself.
        canvas (array): if the canvas doesn't change it can be faster to re-use it. Optional.
        zplane (float): a plane of depth to solve intersection when
            vertex points project behind the camera plane. 
    """
    if R is None:
        R = util.euler2mat([np.pi/3, 0, 0])

    if mode == '2D_only':
        
        im_drawn_rgb = deepcopy(im)

        # go in order of reverse depth
        for mesh_idx in reversed(np.argsort([mesh.verts_padded().cpu().mean(1)[0, 1] for mesh in meshes])):
            mesh = meshes[mesh_idx]

            verts3D = mesh.verts_padded()[0].numpy()
            verts2D = (K @ verts3D.T) / verts3D[:, -1]

            color = [min(255, c*255*1.25) for c in mesh.textures.verts_features_padded()[0,0].tolist()]
            
            x1 = verts2D[0, :].min()
            y1 = verts2D[1, :].min() 
            x2 = verts2D[0, :].max() 
            y2 = verts2D[1, :].max() 

            draw_2d_box(im_drawn_rgb, [x1, y1, x2-x1, y2-y1], color=color, thickness=max(2, int(np.round(3*im_drawn_rgb.shape[0]/1250))))
            
            if text is not None:
                draw_text(im_drawn_rgb, '{}'.format(text[mesh_idx]), [x1, y1], scale=0.50*im_drawn_rgb.shape[0]/500, bg_color=color)
        
        return im_drawn_rgb

    else:
        
        meshes_scene = join_meshes_as_scene(meshes).cuda()
        device = meshes_scene.device
        meshes_scene.textures = meshes_scene.textures.to(device)

        cameras = util.get_camera(K, im.shape[1], im.shape[0]).to(device)
        renderer = util.get_basic_renderer(cameras, im.shape[1], im.shape[0], use_color=True).to(device)
        

        if mode in ['front_and_novel', 'front']:
            '''
            Render full scene from image view
            '''
            
            im_drawn_rgb = deepcopy(im)

            # save memory if not blending the render
            if blend_weight > 0:
                rendered_img, _ = renderer(meshes_scene)
                sil_mask = rendered_img[0, :, :, 3].cpu().numpy() > 0.1
                rendered_img = (rendered_img[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)    
                im_drawn_rgb[sil_mask] = rendered_img[sil_mask] * blend_weight + im_drawn_rgb[sil_mask] * (1 - blend_weight)

            '''
            Draw edges for image view
            '''
            
            # go in order of reverse depth
            for mesh_idx in reversed(np.argsort([mesh.verts_padded().cpu().mean(1)[0, 1] for mesh in meshes])):
                mesh = meshes[mesh_idx]

                verts3D = mesh.verts_padded()[0].cpu().numpy()
                verts2D = (K @ verts3D.T) / verts3D[:, -1]

                color = [min(255, c*255*1.25) for c in mesh.textures.verts_features_padded()[0,0].tolist()]

                draw_3d_box_from_verts(
                    im_drawn_rgb, K, verts3D, color=color, 
                    thickness=max(2, int(np.round(3*im_drawn_rgb.shape[0]/1250))), 
                    draw_back=False, draw_top=False, zplane=zplane
                )

                x1 = verts2D[0, :].min() #min(verts2D[0, (verts2D[0, :] > 0) & (verts2D[0, :] < im_drawn_rgb.shape[1])])
                y1 = verts2D[1, :].min() #min(verts2D[1, (verts2D[1, :] > 0) & (verts2D[1, :] < im_drawn_rgb.shape[0])])
                
                if text is not None:
                    draw_text(im_drawn_rgb, '{}'.format(text[mesh_idx]), [x1, y1], scale=0.50*im_drawn_rgb.shape[0]/500, bg_color=color)

            if blend_weight_overlay < 1.0 and blend_weight_overlay > 0.0:
                im_drawn_rgb = im_drawn_rgb * blend_weight_overlay + deepcopy(im) * (1 - blend_weight_overlay)

        if mode == 'front':
            return im_drawn_rgb

        elif mode in ['front_and_novel', 'novel']:

            '''
            Render from a new view
            '''
            
            has_canvas_already = canvas is not None
            if not has_canvas_already:
                canvas = np.ones((scale, scale, 3))

            view_R = torch.from_numpy(R).float().to(device)

            if T is None:
                center = (meshes_scene.verts_padded().min(1).values + meshes_scene.verts_padded().max(1).values).unsqueeze(0)/2
            else:
                center = torch.from_numpy(T).float().to(device).view(1, 1, 3)
            
            verts_rotated = meshes_scene.verts_padded().clone()
            verts_rotated -= center
            verts_rotated = (view_R @ verts_rotated[0].T).T.unsqueeze(0)

            K_novelview = deepcopy(K)
            K_novelview[0, -1] *= scale / im.shape[1]
            K_novelview[1, -1] *= scale / im.shape[0]

            cameras = util.get_camera(K_novelview, scale, scale).to(device)
            renderer = util.get_basic_renderer(cameras, scale, scale, use_color=True).to(device)

            margin = 0.01

            if T is None:
                max_trials = 10000
                zoom_factor = 100.0
                zoom_factor_in = zoom_factor

                while max_trials:
                    zoom_factor_in = zoom_factor_in*0.95
                    verts = verts_rotated.clone()
                    verts[:, :, -1] += center[:, :, -1]*zoom_factor_in
                    verts_np = verts.cpu().numpy()

                    proj = ((K_novelview @ verts_np[0].T) / verts_np[:, :, -1])

                    # some vertices are extremely close or negative...
                    # this implies we have zoomed in too much
                    if (verts[0, :, -1] < 0.25).any():
                        break
                    
                    # left or above image
                    elif (proj[:2, :] < scale*margin).any():
                        break
                    
                    # right or below borders
                    elif (proj[:2, :] > scale*(1 - margin)).any():
                        break

                    # everything is in view.
                    zoom_factor = zoom_factor_in
                    max_trials -= 1

                zoom_out_bias = center[:, :, -1].item()
            else:
                zoom_out_bias = 1.0

            verts_rotated[:, :, -1] += zoom_out_bias*zoom_factor
            meshes_novel_view = meshes_scene.clone().update_padded(verts_rotated)

            rendered_img, _ = renderer(meshes_novel_view)
            im_novel_view = (rendered_img[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
            sil_mask = rendered_img[0, :, :, 3].cpu().numpy() > 0.1
            
            center_np = center.cpu().numpy()
            view_R_np = view_R.cpu().numpy()

            if not has_canvas_already:
                if ground_bounds is None:

                    min_x3d, _, min_z3d = meshes_scene.verts_padded().min(1).values[0, :].tolist()
                    max_x3d, max_y3d, max_z3d = meshes_scene.verts_padded().max(1).values[0, :].tolist()

                    # go for grid projection, but with extremely bad guess at bounds
                    x3d_start = np.round(min_x3d - (max_x3d - min_x3d)*50)
                    x3d_end = np.round(max_x3d + (max_x3d - min_x3d)*50)
                    z3d_start = np.round(min_z3d - (max_z3d - min_z3d)*50)
                    z3d_end = np.round(max_z3d + (max_z3d - min_z3d)*50)

                    grid_xs = np.arange(x3d_start, x3d_end)
                    grid_zs = np.arange(z3d_start, z3d_end)

                    xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
                    ys_mesh = np.ones_like(xs_mesh)*max_y3d

                    point_mesh = np.concatenate((xs_mesh[:, :, np.newaxis], ys_mesh[:, :, np.newaxis], zs_mesh[:, :, np.newaxis]), axis=2)
                    point_mesh_orig = deepcopy(point_mesh)

                    mesh_shape = point_mesh.shape
                    point_mesh = view_R_np @ (point_mesh - center_np).transpose(2, 0, 1).reshape(3, -1)
                    point_mesh[-1] += zoom_out_bias*zoom_factor
                    point_mesh[-1, :] = point_mesh[-1, :].clip(0.25)
                    point_mesh_2D = (K_novelview @ point_mesh) / point_mesh[-1]
                    point_mesh_2D[-1] = point_mesh[-1]

                    point_mesh = point_mesh.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)
                    point_mesh_2D = point_mesh_2D.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)

                    maskx = (point_mesh_2D[:, :, 0].T >= -50) & (point_mesh_2D[:, :, 0].T < scale+50) & (point_mesh_2D[:, :, 2].T > 0)
                    maskz = (point_mesh_2D[:, :, 1].T >= -50) & (point_mesh_2D[:, :, 1].T < scale+50) & (point_mesh_2D[:, :, 2].T > 0)

                    # invalid scene?
                    if (not maskz.any()) or (not maskx.any()):
                        return im, im, canvas

                    # go for grid projection again!! but with sensible bounds    
                    x3d_start = np.round(point_mesh[:, :, 0].T[maskx].min() - 10)
                    x3d_end = np.round(point_mesh[:, :, 0].T[maskx].max() + 10)
                    z3d_start = np.round(point_mesh_orig[:, :, 2].T[maskz].min() - 10)
                    z3d_end = np.round(point_mesh_orig[:, :, 2].T[maskz].max() + 10)

                else:
                    max_y3d, x3d_start, x3d_end, z3d_start, z3d_end = ground_bounds

                grid_xs = np.arange(x3d_start, x3d_end)
                grid_zs = np.arange(z3d_start, z3d_end)

                xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
                ys_mesh = np.ones_like(xs_mesh)*max_y3d

                point_mesh = np.concatenate((xs_mesh[:, :, np.newaxis], ys_mesh[:, :, np.newaxis], zs_mesh[:, :, np.newaxis]), axis=2)

                mesh_shape = point_mesh.shape
                point_mesh = view_R_np @ (point_mesh - center_np).transpose(2, 0, 1).reshape(3, -1)
                point_mesh[-1] += zoom_out_bias*zoom_factor
                point_mesh[-1, :] = point_mesh[-1, :].clip(0.25)
                point_mesh_2D = (K_novelview @ point_mesh) / point_mesh[-1]
                point_mesh_2D[-1] = point_mesh[-1]

                point_mesh = point_mesh.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)
                point_mesh_2D = point_mesh_2D.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)

                bg_color = (225,)*3
                line_color = (175,)*3
                canvas[:, :, 0] = bg_color[0]
                canvas[:, :, 1] = bg_color[1]
                canvas[:, :, 2] = bg_color[2]
                lines_to_draw = set()

                for grid_row_idx in range(1, len(grid_zs)):

                    pre_z = grid_zs[grid_row_idx-1]
                    cur_z = grid_zs[grid_row_idx]

                    for grid_col_idx in range(1, len(grid_xs)):
                        pre_x = grid_xs[grid_col_idx-1]
                        cur_x = grid_xs[grid_col_idx]
                        
                        p1 = point_mesh_2D[grid_row_idx-1, grid_col_idx-1]
                        valid1 = p1[-1] > 0
                        p2 = point_mesh_2D[grid_row_idx-1, grid_col_idx]
                        valid2 = p2[-1] > 0
                        if valid1 and valid2:
                            line = (tuple(p1[:2].astype(int).tolist()), tuple(p2[:2].astype(int).tolist()))
                            lines_to_draw.add(line)

                        # draw vertical line from the previous row
                        p1 = point_mesh_2D[grid_row_idx-1, grid_col_idx-1]
                        valid1 = p1[-1] > 0
                        p2 = point_mesh_2D[grid_row_idx, grid_col_idx-1]
                        valid2 = p2[-1] > 0
                        if valid1 and valid2:
                            line = (tuple(p1[:2].astype(int).tolist()), tuple(p2[:2].astype(int).tolist()))
                            lines_to_draw.add(line)

                for line in lines_to_draw:
                    draw_line(canvas, line[0], line[1], color=line_color, thickness=max(1, int(np.round(3*scale/1250))))

            im_novel_view[~sil_mask] = canvas[~sil_mask]

            '''
            Draw edges for novel view
            '''

            # apply novel view to meshes
            meshes_novel = []

            for mesh in meshes:
                
                mesh_novel = mesh.clone().to(device)

                verts_rotated = mesh_novel.verts_padded()
                verts_rotated -= center
                verts_rotated = (view_R @ verts_rotated[0].T).T.unsqueeze(0)
                verts_rotated[:, :, -1] += zoom_out_bias*zoom_factor
                mesh_novel = mesh_novel.update_padded(verts_rotated)

                meshes_novel.append(mesh_novel)

            # go in order of reverse depth
            for mesh_idx in reversed(np.argsort([mesh.verts_padded().cpu().mean(1)[0, 1] for mesh in meshes_novel])):
                mesh = meshes_novel[mesh_idx]

                verts3D = mesh.verts_padded()[0].cpu().numpy()
                verts2D = (K_novelview @ verts3D.T) / verts3D[:, -1]

                color = [min(255, c*255*1.25) for c in mesh.textures.verts_features_padded()[0,0].tolist()]

                draw_3d_box_from_verts(
                    im_novel_view, K_novelview, verts3D, color=color, 
                    thickness=max(2, int(np.round(3*im_novel_view.shape[0]/1250))), 
                    draw_back=False, draw_top=False, zplane=zplane
                )
                
                x1 = verts2D[0, :].min() 
                y1 = verts2D[1, :].min() 
                
                if text is not None:
                    draw_text(im_novel_view, '{}'.format(text[mesh_idx]), [x1, y1], scale=0.50*im_novel_view.shape[0]/500, bg_color=color)

            if mode == 'front_and_novel':
                return im_drawn_rgb, im_novel_view, canvas
            else:
                return im_novel_view, canvas

        else:
            raise ValueError('No visualization written for {}'.format(mode))

def get_polygon_grid(im, poly_verts):

    nx = im.shape[1]
    ny = im.shape[0]

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid

def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):

    if fill: thickness = -1

    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)

def draw_transparent_polygon(im, verts, blend=0.5, color=(0, 255, 255)):

    mask = get_polygon_grid(im, verts[:4, :])

    im[mask, 0] = im[mask, 0] * blend + (1 - blend) * color[0]
    im[mask, 1] = im[mask, 1] * blend + (1 - blend) * color[1]
    im[mask, 2] = im[mask, 2] * blend + (1 - blend) * color[2]


def draw_3d_box_from_verts(im, K, verts3d, color=(0, 200, 200), thickness=1, draw_back=False, draw_top=False, zplane=0.05, eps=1e-4):
    """
    Draws a scene from multiple different modes. 
    Args:
        im (array): the image to draw onto
        K (array): the 3x3 matrix for projection to camera to screen
        verts3d (array): the 8x3 matrix of vertices in camera space
        color (tuple): color in RGB scaled [0, 255)
        thickness (float): the line thickness for opencv lines
        draw_back (bool): whether a backface should be highlighted
        draw_top (bool): whether the top face should be highlighted
        zplane (float): a plane of depth to solve intersection when
            vertex points project behind the camera plane. 
    """

    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()

    if isinstance(verts3d, torch.Tensor):
        verts3d = verts3d.detach().cpu().numpy()

    # reorder
    bb3d_lines_verts = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 5], [5, 6], [6, 2], [4, 5], [4, 7], [6, 7], [0, 4], [3, 7]]
    
    # define back and top vetice planes
    back_idxs = [4, 0, 3, 7]
    top_idxs = [4, 0, 1, 5]
    
    for (i, j) in bb3d_lines_verts:
        v0 = verts3d[i]
        v1 = verts3d[j]

        z0, z1 = v0[-1], v1[-1]

        if (z0 >= zplane or z1 >= zplane):
            
            # computer intersection of v0, v1 and zplane
            s = (zplane - z0) / max((z1 - z0), eps)
            new_v = v0 + s * (v1 - v0)

            if (z0 < zplane) and (z1 >= zplane):
                # i0 vertex is behind the plane
                v0 = new_v
            elif (z0 >= zplane) and (z1 < zplane):
                # i1 vertex is behind the plane
                v1 = new_v

            v0_proj = (K @ v0)/max(v0[-1], eps)
            v1_proj = (K @ v1)/max(v1[-1], eps)

            # project vertices
            cv2.line(im, 
                (int(v0_proj[0]), int(v0_proj[1])), 
                (int(v1_proj[0]), int(v1_proj[1])), 
                color, thickness
            )

    # dont draw  the planes if a vertex is out of bounds
    draw_back &= np.all(verts3d[back_idxs, -1] >= zplane)
    draw_top &= np.all(verts3d[top_idxs, -1] >= zplane)

    if draw_back or draw_top:
        
        # project to image
        verts2d = (K @ verts3d.T).T
        verts2d /= verts2d[:, -1][:, np.newaxis]
        
        if type(verts2d) == torch.Tensor:
            verts2d = verts2d.detach().cpu().numpy()

        if draw_back:
            draw_transparent_polygon(im, verts2d[back_idxs, :2], blend=0.5, color=color)

        if draw_top:
            draw_transparent_polygon(im, verts2d[top_idxs, :2], blend=0.5, color=color)
    

def draw_3d_box(im, K, box3d, R, color=(0, 200, 200), thickness=1, draw_back=False, draw_top=False, view_R=None, view_T=None):

    verts2d, verts3d = util.get_cuboid_verts(K, box3d, R, view_R=view_R, view_T=view_T)
    draw_3d_box_from_verts(im, K, verts3d, color=color, thickness=thickness, draw_back=draw_back, draw_top=draw_top)

def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):

    text = str(text)
    pos = [int(pos[0]), int(pos[1])]

    if color == 'auto':
        
        if bg_color is not None:
            color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
        else:
            color = (0, 0, 0) 

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)
        
        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)


def draw_transparent_square(im, pos, alpha=1, radius=5, color=(250, 100, 100)):

    l = pos[1] - radius
    r = pos[1] + radius

    t = pos[0] - radius
    b = pos[0] + radius

    if (np.array([l, r, t, b]) >= 0).any():
        l = np.clip(np.floor(l), 0, im.shape[0]).astype(int)
        r = np.clip(np.floor(r), 0, im.shape[0]).astype(int)

        t = np.clip(np.floor(t), 0, im.shape[1]).astype(int)
        b = np.clip(np.floor(b), 0, im.shape[1]).astype(int)

        # blend
        im[l:r + 1, t:b + 1, 0] = im[l:r + 1, t:b + 1, 0] * alpha + color[0] * (1 - alpha)
        im[l:r + 1, t:b + 1, 1] = im[l:r + 1, t:b + 1, 1] * alpha + color[1] * (1 - alpha)
        im[l:r + 1, t:b + 1, 2] = im[l:r + 1, t:b + 1, 2] * alpha + color[2] * (1 - alpha)


def draw_2d_box(im, box, color=(0, 200, 200), thickness=1):

    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x2 = (x + w) - 1
    y2 = (y + h) - 1

    cv2.rectangle(im, (int(x), int(y)), (int(x2), int(y2)), color, thickness)


def imhstack(im1, im2):

    sf = im1.shape[0] / im2.shape[0]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[1] / sf), im1.shape[0]))
    elif sf < 1:
        im1 = cv2.resize(im1, (int(im1.shape[1] / sf), im2.shape[0]))


    im_concat = np.hstack((im1, im2))

    return im_concat


def imvstack(im1, im2):

    sf = im1.shape[1] / im2.shape[1]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[0] / sf), im1.shape[1]))
    elif sf < 1:
        im1 = cv2.resize(im1, (int(im1.shape[0] / sf), im2.shape[1]))

    im_concat = np.vstack((im1, im2))

    return im_concat