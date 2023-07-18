import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    PerspectiveCameras, FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, PointLights, TexturesVertex,
)
import transforms3d as t3d
from HOnnotate_refine.eval.utilsEval import showHandJoints

import chumpy as ch
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model as load_mano_model
import pickle

import modules.config as cfg
from modules.NIA_mano_layer_annotate import Model
import modules.NIA_utils as NIA_utils


DEBUG_IDX = 3

def init_pytorch3d(device=None, camParam=None):
    intrinsic, extrinsic = camParam

    image_size = (cfg.ORIGIN_HEIGHT, cfg.ORIGIN_WIDTH)
    focal_l = (intrinsic[0, 0], intrinsic[1, 1])
    principal_p = (intrinsic[0, -1], intrinsic[1, -1])
    R = torch.unsqueeze(torch.FloatTensor(extrinsic[:, :-1]), 0)
    T = torch.unsqueeze(torch.FloatTensor(extrinsic[:, -1]), 0)

    cameras = PerspectiveCameras(device=device, image_size=(image_size,), focal_length=(focal_l,),
                                 principal_point=(principal_p,), R=R, T=T, in_ndc=False)

    # Set blend params
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # Define the settings for rasterization and shading.

    raster_settings = RasterizationSettings(
        image_size=(cfg.ORIGIN_HEIGHT, cfg.ORIGIN_WIDTH),
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=1,
        bin_size = None,
        max_faces_per_bin = None
    )

    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))

    renderer_rgb = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    rasterizer_depth = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    # silhouette_renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras,
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftSilhouetteShader(blend_params=blend_params)
    # )

    return cameras, blend_params, raster_settings, lights, rasterizer_depth, renderer_rgb


class manoFitter(object):
    def __init__(self, camIDset, camParams, flag_multi=False):
        self.lr_rot_init = 1.0
        self.lr_pose_init = 0.2
        self.lr_xyz_root_init = 1.0
        self.lr_all_init = 0.4
        self.loss_rot_best = float('inf')
        self.loss_pose_best = float('inf')
        self.loss_xyz_root_best = float('inf')
        self.loss_all_best = float('inf')
        self.img_input_height = None
        self.img_input_width = None

        self.device = torch.device("cuda:0")

        # Multi-view setting
        self.camIDset = camIDset
        self.renderer_depth_list = []
        self.renderer_col_list = []

        # Prep pytorch 3d for visualization
        self.intrinsicSet, self.extrinsicSet = camParams

        if not flag_multi:
            camParam = [self.intrinsicSet['mas'], self.extrinsicSet['mas']]

            _, _, _, _, renderer_depth, renderer_col = init_pytorch3d(self.device, camParam)
            self.renderer_depth_list.append(renderer_depth)
            self.renderer_col_list.append(renderer_col)

        else:
            for camID in camIDset:
                camParam = [self.intrinsicSet[camID], self.extrinsicSet[camID]]

                _, _, _, _, renderer_depth, renderer_col = init_pytorch3d(self.device, camParam)
                self.renderer_depth_list.append(renderer_depth)
                self.renderer_col_list.append(renderer_col)

        self.mano_model = Model(mano_path=cfg.MANO_ROOT, device=self.device, batch_size=1, root_idx=0).to(self.device)

        self.mano_model.set_renderer(self.renderer_depth_list[0], self.renderer_col_list[0])
        self.mano_model.change_render_setting(True)
        self.optimizer_adam_mano_fit_all = None
        self.lr_scheduler_all = None

        self.img_input_width = cfg.IMG_WIDTH
        self.img_input_height = cfg.IMG_HEIGHT


    def change_renderer_cam(self, idx):
        camID = self.camIDset[idx]
        self.mano_model.set_renderer(self.renderer_depth_list[idx], self.renderer_col_list[idx])
        self.mano_model.set_cam_params(self.intrinsicSet['mas'], self.extrinsicSet['mas'])

    def get_rendered_img(self, img2bb=None, bbox=None):
        self.mano_model.change_render_setting(True)
        _, _, _, _, depth_rendered, img_rendered, hand_joints = self.mano_model()

        # mano_3Djoint = hand_joints.cpu().data.numpy()[0]

        # transfer to cpu
        img_rendered = (img_rendered.cpu().data.numpy()[0][:,:,:3]*255.0).astype(np.uint8)
        depth_rendered = depth_rendered.cpu().data.numpy()[0]
        kpts_2d = self.mano_model.kpts_2d[0].cpu().data.numpy()

        # if bbox exist, crop rendered mesh image and project 2D keypoints
        if bbox is not None and img2bb is not None:
            # bbox = [x_min, y_min, self.bbox_width, self.bbox_height]
            bbox = bbox.astype(int)
            img_rendered = img_rendered[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
            depth_rendered = depth_rendered[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]

            uv1 = np.concatenate((kpts_2d, np.ones_like(kpts_2d[:, :1])), 1)
            kpts_2d = (img2bb @ uv1.T).T
        img_rendered = NIA_utils.paint_kpts(None, img_rendered, kpts_2d)
        return img_rendered, depth_rendered

    def fit_2d_pose(self, cam, kpts_2d_gt, iter=300):
        intrinsic, extrinsic = cam
        self.mano_model.set_cam_params(intrinsic, extrinsic)
        self.mano_model.set_renderer(self.renderer_depth_list[0], self.renderer_col_list[0])

        self.mano_model.set_kpts_2d_gt(kpts_2d_gt)

        self.mano_model.change_grads(root=True, rot=True, pose=True, shape=False)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose')

        self.fit_mano(optimizer_adam_mano_fit, "all", iter, is_loss_2d=True, \
            is_loss_reg=True, is_debugging=True)
        self.reset_mano_optimization_var()

    def fit_multi2d_pose(self, camSet, rgbSet, depthSet, metas, iter=300):

        self.mano_model.change_grads(root=True, rot=True, pose=True, shape=False)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose')

        self.fit_mano_multiview(camSet, rgbSet, depthSet, metas, optimizer_adam_mano_fit, "all", iter, is_loss_2d=True, \
            is_loss_reg=True, is_loss_depth=False, is_debugging=True)

        self.reset_mano_optimization_var()


    def compute_loss(self, optimizer, mode, is_loss_seg=False, \
                 is_loss_2d=False, is_loss_leap3d=False, is_loss_reg=False, \
                 best_performance=True, is_debugging=True, is_visualizing=False,
                 show_progress=False):
        self.mano_model.set_loss_mode(is_loss_seg=is_loss_seg, is_loss_2d=is_loss_2d,
                                      is_loss_leap3d=is_loss_leap3d, is_loss_reg=is_loss_reg)
        self.mano_model.change_render_setting(False)


    def fit_mano(self, optimizer, mode, iter_fit, is_loss_seg=False, \
                 is_loss_2d=False, is_loss_reg=False, \
                 best_performance=True, is_debugging=True, is_visualizing=False,
                 show_progress=False):

        self.mano_model.set_loss_mode(is_loss_seg=is_loss_seg, is_loss_2d=is_loss_2d, is_loss_reg=is_loss_reg)
        self.mano_model.change_render_setting(False)

        if is_debugging:
            print("Fitting {}".format(mode))
        iter_count = 0
        lr_stage = 3
        update_finished = False

        rot_th = 20000
        pose_th = 20000
        xyz_root_th = 1000

        while not update_finished:
            loss_seg_batch, loss_2d_batch, loss_reg_batch, _, _, _ = self.mano_model()

            loss_total = 0
            loss_seg_sum = 0
            loss_2d_sum = 0
            loss_3d_can_sum = 0
            loss_reg_sum = 0
            loss_leap3d_sum = 0
            if is_loss_seg:
                loss_seg_sum = torch.sum(loss_seg_batch)
                loss_total += loss_seg_sum
            if is_loss_2d:
                loss_2d_sum = torch.sum(loss_2d_batch)
                loss_total += loss_2d_sum
            if is_loss_reg:
                loss_reg_sum = torch.sum(loss_reg_batch)
                loss_total += loss_reg_sum

            if is_debugging and iter_count % 1 == 0:
                print(
                    "[{}] Fit loss total {:.5f}, loss 2d {:.2f}, loss 3d {:.5f}, loss leap3d {:.2f}, loss reg {:.2f}, " \
                    .format(iter_count, float(loss_total), float(loss_2d_sum), float(loss_3d_can_sum), \
                            float(loss_leap3d_sum), float(loss_reg_sum)))

            iter_count += 1

            # Check stopping criteria
            if mode == 'rot':
                if loss_leap3d_sum < rot_th:
                    update_finished = True

            if mode == 'pose':
                if loss_leap3d_sum < pose_th:
                    update_finished = True

            if mode == 'xyz_root':
                if loss_2d_sum < xyz_root_th:
                    update_finished = True

            if mode == 'all':
                if not best_performance:
                    if iter_count >= 10:
                        if loss_2d_sum < 1500:
                            update_finished = True
                    else:
                        if loss_2d_sum < 1000:
                            update_finished = True
                else:
                    if loss_2d_sum < 1000:
                        update_finished = True

            if iter_count >= iter_fit:
                update_finished = True

            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer.step()

            # Adjust pinky
            # with torch.no_grad():
            #     self.mano_model.input_pose[0, cfg.fin4_ver_fix_idx - 3] = \
            #         -self.mano_model.input_pose[0, cfg.fin4_ver_idx2 - 3]

        if is_debugging:
            print("Optimization stopped. Iter = {}".format(iter_count))

        if is_visualizing or show_progress:
            self.mano_model.change_render_setting(True)
            _, _, _, _, _, img_render, _ = self.mano_model()
            img_render = (img_render.cpu().data.numpy()[0][:, :, :3] * 255.0).astype(np.uint8)
            img_render = NIA_utils.paint_kpts(None, img_render, self.mano_model.kpts_2d[0].cpu().data.numpy())
            return img_render
        else:
            self.mano_model()

    def fit_mano_multiview(self, camSet, rgbSet, depthSet, metas, optimizer, mode, iter_fit, is_loss_seg=False, \
                 is_loss_2d=False, is_loss_reg=False, is_loss_depth=True, is_debugging=True, is_visualizing=False, show_progress=False):

        self.mano_model.set_loss_mode(is_loss_seg, is_loss_2d, is_loss_reg, is_loss_depth)
        self.mano_model.change_render_setting(False)

        if is_debugging:
            print("Fitting {}".format(mode))
        iter_count = 0
        lr_stage = 3
        update_finished = False

        # rot_th = 20000
        # pose_th = 20000
        xyz_root_th = 1000

        while not update_finished:

            loss_total = 0
            loss_seg_sum = 0
            loss_2d_sum = 0
            loss_reg_sum = 0
            loss_dep_sum = 0

            for camIdx, camID in enumerate(self.camIDset):
                # if not camIdx == DEBUG_IDX:
                #     continue
                intrinsic, extrinsic = camSet[camIdx]
                self.mano_model.set_cam_params(intrinsic, extrinsic)
                # self.mano_model.set_renderer(self.renderer_depth_list[camIdx], self.renderer_col_list[camIdx])

                meta = metas[camIdx]
                kpts_GT = np.copy(meta['kpts'])
                # check if the view has valid GT value
                if np.isnan(kpts_GT).any():
                    continue

                kpts_2d_gt = kpts_GT[:, :2]
                self.mano_model.set_kpts_2d_gt(kpts_2d_gt)

                depth = depthSet[camIdx] / 10.0
                self.mano_model.set_depthmap(depth)

                bbox = np.copy(meta['bb'])
                self.mano_model.set_bbox(bbox)

                loss_seg_batch, loss_2d_batch, loss_reg_batch, loss_dep_batch, _, _, _ = self.mano_model()

                if is_loss_seg:
                    loss_seg_sum = torch.sum(loss_seg_batch)
                    loss_total += loss_seg_sum
                if is_loss_2d:
                    loss_2d_sum = torch.sum(loss_2d_batch)
                    loss_total += loss_2d_sum
                if is_loss_reg:
                    loss_reg_sum = torch.sum(loss_reg_batch)
                    loss_total += loss_reg_sum
                if is_loss_depth:
                    loss_dep_sum = torch.sum(loss_dep_batch)
                    loss_total += loss_dep_sum

            if is_debugging and iter_count % 1 == 0:
                print(
                    "[{}] Fit loss total {:.5f}, loss seg {:.2f}, loss 2d {:.2f}, loss reg {:.2f}, loss depth {:.2f}, " \
                    .format(iter_count, float(loss_total), float(loss_seg_sum), float(loss_2d_sum), float(loss_reg_sum), float(loss_dep_sum)))

            iter_count += 1

            # Check stopping criteria
            # if mode == 'rot':
            #     if loss_leap3d_sum < rot_th:
            #         update_finished = True
            # if mode == 'pose':
            #     if loss_leap3d_sum < pose_th:
            #         update_finished = True
            if mode == 'xyz_root':
                if loss_2d_sum < xyz_root_th:
                    update_finished = True
            if mode == 'all':
                if loss_2d_sum < 1000:
                    update_finished = True
            if iter_count >= iter_fit:
                update_finished = True

            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer.step()

            # Adjust pinky
            # with torch.no_grad():
            #     self.mano_model.input_pose[0, cfg.fin4_ver_fix_idx - 3] = \
            #         -self.mano_model.input_pose[0, cfg.fin4_ver_idx2 - 3]

        if is_debugging:
            print("Optimization stopped. Iter = {}".format(iter_count))

        if is_visualizing or show_progress:
            self.mano_model.change_render_setting(True)
            _, _, _, _, _, img_render, _ = self.mano_model()
            img_render = (img_render.cpu().data.numpy()[0][:, :, :3] * 255.0).astype(np.uint8)
            img_render = NIA_utils.paint_kpts(None, img_render, self.mano_model.kpts_2d[0].cpu().data.numpy())
            return img_render
        else:
            self.mano_model()


    def reset_mano_optimizer(self, mode):
        if mode == 'rot':
            lr_init = self.lr_rot_init
            model_params = self.mano_model.parameters()
        elif mode == 'pose':
            lr_init = self.lr_pose_init
            model_params = self.mano_model.parameters()
        elif mode == 'xyz_root':
            lr_init = self.lr_xyz_root_init
            params_dict = dict(self.mano_model.named_parameters())
            lr1 = []
            lr2 = []
            for key, value in params_dict.items():
                if value.requires_grad:
                    if 'xy_root' in key:
                        lr1.append(value)
                    elif 'z_root' in key:
                        lr2.append(value)
            model_params = [{'params': lr1, 'lr': lr_init},
                            {'params': lr2, 'lr': lr_init}]
        elif mode == 'all':
            lr_init = self.lr_all_init
            params_dict = dict(self.mano_model.named_parameters())
            lr_xyz_root = []
            lr_rot = []
            lr_pose = []
            for key, value in params_dict.items():
                if value.requires_grad:
                    if 'xy_root' in key:
                        lr_xyz_root.append(value)
                    elif 'z_root' in key:
                        lr_xyz_root.append(value)
                    elif 'input_rot' in key:
                        lr_rot.append(value)
                    elif 'input_pose' in key:
                        lr_pose.append(value)
            model_params = [{'params': lr_xyz_root, 'lr': 0.5},
                            {'params': lr_rot, 'lr': 0.05},
                            {'params': lr_pose, 'lr': 0.05}]
        optimizer_adam_mano_fit = torch.optim.Adam(model_params, lr=lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam_mano_fit, step_size=1, gamma=0.5)
        return optimizer_adam_mano_fit, lr_scheduler

    def reset_mano_optimization_var(self):
        self.mano_model.reset_param_grads()
        self.lr_rot_init = self.lr_rot_init
        self.lr_pose_init = self.lr_pose_init
        self.lr_xyz_root_init = self.lr_xyz_root_init
        self.lr_all_init = self.lr_all_init
        self.loss_rot_best = float('inf')
        self.loss_pose_best = float('inf')
        self.loss_xyz_root_best = float('inf')
        self.loss_all_best = float('inf')

# backup fitting function
"""
    def fit_3d_can_init(self, kpts_3d_can):
        # Set 3D target
        kpts_3d_can = kpts_3d_can.reshape(-1, 3)
        kpts_3d_can -= kpts_3d_can[0]
        kpts_3d_glob = kpts_3d_can * cfg.mano_key_bone_len
        self.mano_model.set_kpts_3d_glob_leap_no_palm(kpts_3d_glob)

        is_debugging = True
        # Step1: fit canonical pose using canonical 3D kpts
        self.mano_model.change_rot_grads(True)
        self.mano_model.set_rot_only()
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('rot')
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "rot", 50, is_loss_leap3d=True, is_optimizing=True, \
                      is_debugging=is_debugging)
        self.reset_mano_optimization_var()

        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose')
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "pose", 50, is_loss_leap3d=True, is_loss_reg=True, \
                      is_optimizing=True, is_debugging=is_debugging)
        self.reset_mano_optimization_var()

    def fit_xyz_root_init(self, kpts_2d_glob):
        # Scale kpts 2d glob to match pytorch 3d resolution
        # assert self.img_input_height is not None
        # kpts_2d_glob = kpts_2d_glob * np.array([float(cfg.IMG_HEIGHT) / self.img_input_height, \
        #                                         float(cfg.IMG_WIDTH) / self.img_input_width])

        is_debugging = True

        # Step2: fit xyz root using global 2D kpts
        self.mano_model.set_xyz_root(torch.tensor([0, 0, 50.0]).view(1, -1).cuda())
        joint_idx_set = set([0, 1, 5, 9, 13, 17])
        for joint_idx, kpt_2d_glob in enumerate(kpts_2d_glob):
            if joint_idx in joint_idx_set:
                self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        self.mano_model.change_root_grads(True)
        optimizer_adam_mano_fit_xyz_root, lr_scheduler = self.reset_mano_optimizer('xyz_root')
        self.fit_mano(optimizer_adam_mano_fit_xyz_root, lr_scheduler, "xyz_root", 50, is_loss_2d_glob=True, \
                      is_optimizing=True, is_debugging=is_debugging)
        self.reset_mano_optimization_var()

    def fit_all_pose(self, kpts_3d_can, kpts_2d_glob):
        # Step final: fit all params using global 2D and canonical 3D kpts
        # Set 3D target
        kpts_3d_can = kpts_3d_can.reshape(-1, 3)
        kpts_3d_can -= kpts_3d_can[0]
        kpts_3d_glob = kpts_3d_can * cfg.mano_key_bone_len
        self.mano_model.set_kpts_3d_glob_leap_no_palm(kpts_3d_glob, with_xyz_root=True)

        is_debugging = True

        for joint_idx, kpt_2d_glob in enumerate(kpts_2d_glob):
            self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        self.mano_model.change_root_grads(True)
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        self.mano_model.change_shape_grads(True)
        if self.optimizer_adam_mano_fit_all is None:
            self.optimizer_adam_mano_fit_all, self.lr_scheduler_all = self.reset_mano_optimizer('all')
        num_iters = 50
        self.fit_mano(self.optimizer_adam_mano_fit_all, self.lr_scheduler_all, "all", num_iters, \
                      is_loss_2d_glob=True, is_loss_leap3d=True, is_loss_reg=True, is_optimizing=True,
                      is_debugging=is_debugging)
        self.reset_mano_optimization_var()
"""



class optimizer_torch():
    def __init__(self, camIDset, camSet, flag_multi=False):
        self.camIDset = camIDset
        if not flag_multi:
            print("initialize optimizer for single-view(master cam)")
            self.mano_fit_tool = manoFitter(camIDset, camSet, flag_multi=flag_multi)
        else:
            print("initialize optimizer for multi-view")
            self.mano_fit_tool = manoFitter(camIDset, camSet, flag_multi=flag_multi)


    def run(self, data):
        camSet, rgbSet, depthSet, metas = data

        # transfer main cam first

        for idx, cam in enumerate(camSet):
            rgbPath = rgbSet[idx]
            depthPath = depthSet[idx]
            meta = metas[idx]

            bbox = np.copy(meta['bb'])
            img2bb = np.copy(meta['img2bb'])
            mp_GT = np.copy(meta['kpts'])
            kpts_3d_gt = mp_GT
            kpts_2d_gt = mp_GT[:, :2]

            self.mano_fit_tool.fit_2d_pose(cam, kpts_2d_gt, iter=500)

            rgb, depth = self.mano_fit_tool.get_rendered_img(img2bb=img2bb, bbox=bbox)
            cv2.imshow("depth", depth)

            # Display blended image
            img_1 = np.copy(rgb)
            img_2 = np.asarray(cv2.imread(rgbPath))
            img_3 = cv2.addWeighted(img_1, 0.5, img_2, 0.7, 0)
            cv2.imshow("debug", img_3)
            cv2.waitKey(0)

            print("only master cam")
            break

    def run_multiview(self, data):
        camSet, rgbSet, depthSet, metas = data

        # fitting mesh
        self.mano_fit_tool.fit_multi2d_pose(camSet, rgbSet, depthSet, metas, iter=700)

        # visualization of each cam results
        for idx, (rgb, depth, meta) in enumerate(zip(rgbSet, depthSet, metas)):
            # if not idx == DEBUG_IDX:
            #     continue
            bbox = np.copy(meta['bb'])
            img2bb = np.copy(meta['img2bb'])
            kpts_2d_gt = np.copy(meta['kpts'])[:, :2]

            # depth = depth / 10.0
            # cv2.imshow("GT_depth", depth)

            # change renderer cam of hand model
            self.mano_fit_tool.change_renderer_cam(idx)

            rgb_mano, depth_mano = self.mano_fit_tool.get_rendered_img(img2bb=img2bb, bbox=bbox)
            # Display blended image
            rgb_blend = cv2.addWeighted(rgb, 0.5, rgb_mano, 0.7, 0)

            # GT 2D pose (mediapipe)
            uv1 = np.concatenate((kpts_2d_gt, np.ones_like(kpts_2d_gt[:, :1])), 1)
            kpts_2d_gt = (img2bb @ uv1.T).T
            rgb_GT = NIA_utils.paint_kpts(None, np.copy(rgb), kpts_2d_gt)

            imgName_blend = "Pred_rgb_" + str(self.camIDset[idx])
            imgName_GT = "GT_rgb_" + str(self.camIDset[idx])
            imgName_depth = "Pred_depth_" + str(self.camIDset[idx])
            cv2.imshow(imgName_blend, rgb_blend)
            cv2.imshow(imgName_GT, rgb_GT)
            cv2.imshow(imgName_depth, depth_mano)
            cv2.waitKey(0)




if __name__ == '__main__':
   print("l")