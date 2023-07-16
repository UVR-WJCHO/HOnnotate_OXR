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


def init_pytorch3d(camParam=None):
    device = torch.device("cuda:0")

    ### debugging Perspective Camera ###
    # intrinsic, extrinsic = camParam
    # K = np.eye(4)
    # K[:-1, :-1] = intrinsic
    # K = torch.unsqueeze(torch.FloatTensor(K), 0)
    # R = torch.unsqueeze(torch.FloatTensor(extrinsic[:, :-1]), 0)
    # T = torch.unsqueeze(torch.FloatTensor(extrinsic[:, -1]), 0)
    #
    # img_size = np.zeros((2))
    # img_size[0] = cfg.IMG_HEIGHT
    # img_size[1] = cfg.IMG_WIDTH
    # img_size = torch.unsqueeze(torch.FloatTensor(img_size), 0)
    #
    # cameras = PerspectiveCameras(R=R, T=T, K=K, device=device,image_size=img_size)

    ### Manual fov calculation. wrong ###
    w, h = 1920, 1080
    fx, fy = 908.67419434, 908.34960938
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    fov = (fov_y + fov_x) / 2.0

    # Initialize an OpenGL perspective camera.
    cameras = FoVPerspectiveCameras(fov=fov, device=device)

    # Set blend params
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # Define the settings for rasterization and shading.

    raster_settings = RasterizationSettings(
        image_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=20,
        bin_size = None,
        max_faces_per_bin = None
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    rasterizer_col = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    return device, cameras, blend_params, raster_settings, rasterizer_col, lights, phong_renderer


def load_mano_model(mano_path, silhouette_renderer, phong_renderer, device):
    root_idx = 0
    mano_model = Model(
        mano_path = mano_path,
        renderer = silhouette_renderer,
        renderer2 = phong_renderer,
        device = device,
        batch_size = 1,
        root_idx = root_idx).to(device)
    return mano_model

class manoFitter(object):
    def __init__(self, camParam):


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

        # Prep pytorch 3d for visualization
        self.device, self.cameras, self.blend_params, self.raster_settings, self.silhouette_renderer, \
            self.lights, self.phong_renderer = init_pytorch3d(camParam)
        self.mano_model = load_mano_model(cfg.MANO_ROOT, self.silhouette_renderer, self.phong_renderer, self.device)

        self.mano_model.change_render_setting(True)
        self.optimizer_adam_mano_fit_all = None
        self.lr_scheduler_all = None

        self.img_input_width = cfg.IMG_WIDTH
        self.img_input_height = cfg.IMG_HEIGHT

    def get_rendered_img(self, img2bb=None):
        self.mano_model.change_render_setting(True)
        _, _, _, _, depth_rendered, img_rendered = self.mano_model()
        img_rendered = (img_rendered.cpu().data.numpy()[0][:,:,:3]*255.0).astype(np.uint8)
        depth_rendered = (depth_rendered.cpu().data.numpy()[0][:, :, :3] * 255.0).astype(np.uint8)

        kpts_2d = self.mano_model.kpts_2d_glob[0].cpu().data.numpy()
        if img2bb is not None:
            uv1 = np.concatenate((kpts_2d, np.ones_like(kpts_2d[:, :1])), 1)
            kpts_2d = (img2bb @ uv1.T).T
        img_rendered = NIA_utils.paint_kpts(None, img_rendered, kpts_2d)
        return img_rendered, depth_rendered

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

    def fit_2d_pose(self, kpts_2d_gt):
        self.mano_model.set_kpts_2d_gt(kpts_2d_gt)

        self.mano_model.change_root_grads(True)
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose')
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "all", 300, is_loss_2d_glob=True, \
            is_loss_reg=True, is_optimizing=True, is_debugging=True)
        self.reset_mano_optimization_var()


    def fit_mano(self, optimizer, lr_scheduler, mode, iter_fit, is_loss_seg=False, \
                 is_loss_2d_glob=False, is_loss_leap3d=False, is_loss_reg=False, is_optimizing=True, \
                 best_performance=True, is_debugging=True, is_visualizing=False,
                 show_progress=False):
        self.mano_model.set_loss_mode(is_loss_seg=is_loss_seg, is_loss_2d_glob=is_loss_2d_glob,
                                      is_loss_leap3d=is_loss_leap3d, is_loss_reg=is_loss_reg)
        self.mano_model.change_render_setting(False)

        if is_debugging:
            print("Fitting {}".format(mode))
        iter_count = 0
        lr_stage = 3
        update_finished = False

        if not best_performance:
            rot_th = 20000
            pose_th = 20000
            xyz_root_th = 2000
        else:
            rot_th = 20000
            pose_th = 20000
            xyz_root_th = 1000

        while not update_finished and is_optimizing:
            loss_seg_batch, loss_2d_glob_batch, loss_reg_batch, loss_leap3d_batch, _, _ = self.mano_model()

            loss_total = 0
            loss_seg_sum = 0
            loss_2d_glob_sum = 0
            loss_3d_can_sum = 0
            loss_reg_sum = 0
            loss_leap3d_sum = 0
            if is_loss_seg:
                loss_seg_sum = torch.sum(loss_seg_batch)
                loss_total += loss_seg_sum
            if is_loss_2d_glob:
                loss_2d_glob_sum = torch.sum(loss_2d_glob_batch)
                loss_total += loss_2d_glob_sum
            if is_loss_reg:
                loss_reg_sum = torch.sum(loss_reg_batch)
                loss_total += loss_reg_sum
            if is_loss_leap3d:
                loss_leap3d_sum = 10000 * torch.sum(loss_leap3d_batch)  # original 100000
                loss_total += loss_leap3d_sum

            if is_debugging and iter_count % 1 == 0:
                print(
                    "[{}] Fit loss total {:.5f}, loss 2d {:.2f}, loss 3d {:.5f}, loss leap3d {:.2f}, loss reg {:.2f}, " \
                    .format(iter_count, float(loss_total), float(loss_2d_glob_sum), float(loss_3d_can_sum), \
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
                if loss_2d_glob_sum < xyz_root_th:
                    update_finished = True

            if mode == 'all':
                if not best_performance:
                    if iter_count >= 10:
                        if loss_2d_glob_sum < 1500:
                            update_finished = True
                    else:
                        if loss_2d_glob_sum < 1000:
                            update_finished = True
                else:
                    if loss_2d_glob_sum < 1000:
                        update_finished = True

            if iter_count >= iter_fit:
                update_finished = True

            if is_optimizing:
                optimizer.zero_grad()
                loss_total.backward(retain_graph=True)
                optimizer.step()

                # Adjust pinky
                # with torch.no_grad():
                #     self.mano_model.input_pose[0, cfg.fin4_ver_fix_idx - 3] = \
                #         -self.mano_model.input_pose[0, cfg.fin4_ver_idx2 - 3]
            else:
                update_finished = True

        if is_debugging:
            print("Optimization stopped. Iter = {}".format(iter_count))

        if is_visualizing or show_progress:
            self.mano_model.change_render_setting(True)
            _, _, _, _, _, img_render = self.mano_model()
            img_render = (img_render.cpu().data.numpy()[0][:, :, :3] * 255.0).astype(np.uint8)
            img_render = NIA_utils.paint_kpts(None, img_render, self.mano_model.kpts_2d_glob[0].cpu().data.numpy())
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

class optimizer_torch():
    def __init__(self):
        from modules.HandsForAll.kpts_global_projector import kpts_global_projector
        self.kpts_global_project_tool = kpts_global_projector('mano_hand', foc_l=908.67, cam_h=cfg.IMG_HEIGHT, cam_w=cfg.IMG_WIDTH)
    def run(self, data):
        camSet, rgbSet, depthSet, metas = data


        # transfer main cam first
        self.mano_fit_tool = manoFitter(camSet[0])




        for idx, cam in enumerate(camSet):
            intrinsic, extrinsic = cam

            self.mano_fit_tool.mano_model.set_cam_intrinsic(intrinsic)
            self.mano_fit_tool.mano_model.set_up_camera_wRT(extrinsic)

            rgbPath = rgbSet[idx]
            depthPath = depthSet[idx]
            meta = metas[idx]

            img2bb = np.copy(meta['img2bb'])
            mp_GT = np.copy(meta['kpts'])
            kpts_3d_gt = mp_GT
            kpts_2d_gt = mp_GT[:, :2]


            self.mano_fit_tool.fit_2d_pose(kpts_2d_gt)

            #### debug
            #
            # img_debug = NIA_utils.paint_kpts(None, rgb, kpts_2d_glob_gt_np)
            # cv2.imshow("debug gt kpts", img_debug)
            # cv2.waitKey(0)

            # # # Fit xyz root
            # self.mano_fit_tool.mano_model.set_xyz_root(torch.from_numpy(kpts_3d_can_pred_np[0, :]).cuda())
            #
            #
            # # Fit pose
            # self.mano_fit_tool.fit_all_pose(kpts_3d_can_pred_np, kpts_2d_glob_gt_np)


            rgb, depth = self.mano_fit_tool.get_rendered_img(img2bb=None)
            cv2.imshow("depth", depth)

            # Display blended image
            img_1 = np.copy(rgb)
            img_2 = np.asarray(cv2.imread(rgbPath))
            img_3 = cv2.addWeighted(img_1, 0.5, img_2, 0.7, 0)
            cv2.imshow("debug", img_3)
            cv2.waitKey(0)

            print("only master cam")
            break

if __name__ == '__main__':
   print("l")