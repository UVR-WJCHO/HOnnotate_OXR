import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../'))

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.lossUtils import *
from utils.modelUtils import *


class MultiViewLossFunc(nn.Module):
    def __init__(self, device='cpu', bs=1, dataloaders=None, renderers=None, losses=None):
        super(MultiViewLossFunc, self).__init__()
        self.device = device
        self.bs = bs
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

        self.pose_reg_tensor, self.pose_mean_tensor = self.get_pose_constraint_tensor()

        self.dataloaders = dataloaders
        self.renderers = renderers

        self.loss_dict = losses

    def set_object_main_extrinsic(self, obj_main_cam_idx):
        cam_params = self.dataloaders[obj_main_cam_idx].cam_parameter
        self.main_Ms_obj = torch.FloatTensor(cam_params[1]).to(self.device)

    def get_pose_constraint_tensor(self):
        pose_mean_tensor = torch.tensor(params.pose_mean_list).cuda()
        pose_reg_tensor = torch.tensor(params.pose_reg_list).cuda()
        return pose_reg_tensor, pose_mean_tensor

    def compute_reg_loss(self, mano_tensor, pose_mean_tensor, pose_reg_tensor):
        reg_loss = ((mano_tensor - pose_mean_tensor) ** 2) * pose_reg_tensor
        return torch.sum(reg_loss, -1)

    def set_gt(self, camIdx, frame):
        gt_sample = self.dataloaders[camIdx][frame]

        self.bb = np.asarray(gt_sample['bb']).astype(int)
        self.img2bb = gt_sample['img2bb']
        self.gt_kpts2d = torch.unsqueeze(torch.FloatTensor(gt_sample['kpts2d']), 0).to(self.device)
        self.gt_kpts3d = torch.unsqueeze(torch.FloatTensor(gt_sample['kpts3d']), 0).to(self.device)

        self.gt_rgb = torch.FloatTensor(gt_sample['rgb']).to(self.device)
        self.gt_depth = torch.unsqueeze(torch.FloatTensor(gt_sample['depth']).to(self.device), 0).to(self.device)
        self.gt_seg = torch.unsqueeze(torch.FloatTensor(gt_sample['seg']).to(self.device), 0).to(self.device)

        # set object seg gt
        # obj_mask = gt_sample['seg_bg'] - gt_sample['seg']
        self.gt_seg_obj = torch.unsqueeze(torch.FloatTensor(gt_sample['seg_obj']).to(self.device), 0).to(self.device)

        # seg_obj = (gt_sample['seg_obj'] * 255.0).astype(np.uint8)
        # cv2.imshow("seg_gap_obj", seg_obj)
        # cv2.waitKey(0)


    def set_cam(self, camIdx):
        cam_params = self.dataloaders[camIdx].cam_parameter
        cam_renderer = self.renderers[camIdx]

        Ks, Ms, _ = cam_params
        self.Ks = torch.FloatTensor(Ks).to(self.device)
        self.Ms = torch.FloatTensor(Ms).to(self.device)

        self.cam_renderer = cam_renderer

    def set_main_cam(self, main_cam_idx=0):
        main_cam_params = self.dataloaders[main_cam_idx].cam_parameter

        main_Ks, main_Ms, _ = main_cam_params
        self.main_Ks = torch.FloatTensor(main_Ks).to(self.device)
        self.main_Ms = torch.FloatTensor(main_Ms).to(self.device)

    def forward(self, pred, pred_obj, render, camIdx, frame):
        # set gt data of current index & camera status
        self.set_gt(camIdx, frame)
        self.set_cam(camIdx)

        verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms, self.main_Ms), 0)
        joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms, self.main_Ms), 0)

        loss = {}
        if 'kpts2d' in self.loss_dict:
            pred_kpts2d = projectPoints(joints_cam, self.Ks)

            # loss_kpts2d = self.mse_loss(pred_kpts2d, self.gt_kpts2d.repeat(self.bs, 1, 1).to(self.device))
            loss_kpts2d = torch.sum(((pred_kpts2d - self.gt_kpts2d) ** 2).reshape(self.bs, -1), -1)
            loss['kpts2d'] = loss_kpts2d

        if 'reg' in self.loss_dict:
            pose_reg = self.compute_reg_loss(pred['pose'], self.pose_mean_tensor, self.pose_reg_tensor)
            shape_reg = torch.sum(
                ((pred['shape'] - torch.zeros_like(pred['shape'])) ** 2).view(self.bs, -1), -1)
            loss_reg = pose_reg + shape_reg
            loss['reg'] = loss_reg

        if render:
            pred_rendered = self.cam_renderer.render(verts_cam, pred['faces'])

            # rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
            # cv2.imshow("rgb_mesh_hand", rgb_mesh)
            # cv2.waitKey(1)

            # TODO : need to combine both verts of hand/object
            if pred_obj is not None:
                verts_obj_cam = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms, self.main_Ms_obj), 0)
                pred_obj_rendered = self.cam_renderer.render(verts_obj_cam, pred_obj['faces'])

                # rgb_mesh_obj = np.squeeze((pred_obj_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                # cv2.imshow("rgb_mesh_obj", rgb_mesh_obj)
                # cv2.waitKey(1)

            if 'seg' in self.loss_dict:
                pred_seg = pred_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                seg_gap = ((pred_seg - self.gt_seg) * pred_seg) ** 2.

                # loss_seg = self.mse_loss(pred_seg, self.gt_seg)
                loss_seg = torch.sum(seg_gap.view(self.bs, -1), -1)
                loss_seg = torch.clamp(loss_seg, min=0, max=5000)  # loss clipping following HOnnotate
                loss['seg'] = loss_seg

                # pred_seg = np.squeeze((pred_seg[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                # cv2.imshow("pred_seg", pred_seg)
                # cv2.waitKey(1)

                if pred_obj is not None:
                    pred_seg_obj = pred_obj_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0]+self.bb[2]]
                    seg_obj_gap = ((pred_seg_obj - self.gt_seg_obj) * pred_seg_obj) ** 2.

                    loss_seg_obj = torch.sum(seg_obj_gap.view(self.bs, -1), -1)
                    loss_seg_obj = torch.clamp(loss_seg_obj, min=0, max=5000)  # loss clipping following HOnnotate
                    loss['seg'] += loss_seg_obj

                    # seg_obj_gap = np.squeeze((seg_obj_gap[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                    # cv2.imshow("seg_obj_gap", seg_obj_gap)
                    # # gt_seg_obj = np.squeeze((self.gt_seg_obj[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                    # # cv2.imshow("gt_seg_obj", gt_seg_obj)
                    # cv2.waitKey(1)

            if 'depth' in self.loss_dict:
                pred_seg = pred_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                pred_depth = pred_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                depth_gap = torch.abs(pred_depth[pred_depth != 0] - self.gt_depth[pred_depth != 0])# * pred_seg

                pred_depth_vis = np.squeeze((pred_depth[0].cpu().detach().numpy())).astype(np.uint8)
                gt_depth_vis = np.squeeze((self.gt_depth[0].cpu().detach().numpy())).astype(np.uint8)
                cv2.imshow("pred_depth", pred_depth_vis)
                cv2.imshow("gt_depth_vis", gt_depth_vis)
                cv2.waitKey(1)

                # loss_depth = self.mse_loss(pred_depth, self.gt_depth)
                loss_depth = torch.sum(depth_gap.view(self.bs, -1), -1)
                loss_depth = torch.clamp(loss_depth, min=0, max=5000)  # loss clipping used in HOnnotate
                loss['depth'] = loss_depth

                if pred_obj is not None:
                    pred_seg_obj = pred_obj_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                    pred_depth_obj = pred_obj_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                    depth_obj_gap = torch.abs(pred_depth_obj - self.gt_depth) * pred_seg_obj

                    loss_depth_obj = torch.sum(depth_obj_gap.view(self.bs, -1), -1)
                    loss_depth_obj = torch.clamp(loss_depth_obj, min=0, max=5000)  # loss clipping used in HOnnotate
                    loss['depth'] += loss_depth_obj

        return loss

    def visualize(self, pred, pred_obj, camIdx, frame, save_path, camID, flag_obj=False, flag_crop=False):
        # set gt to load original input
        self.set_gt(camIdx, frame)
        # set camera status for projection
        self.set_cam(camIdx)

        ## HAND ##
        # project hand joint
        joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms, self.main_Ms), 0)
        pred_kpts2d = projectPoints(joints_cam, self.Ks)

        # render mesh on current cam
        verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms, self.main_Ms), 0)
        pred_rendered = self.cam_renderer.render(verts_cam, pred['faces'])

        ## OBJECT ##
        if flag_obj:
            # if flag_obj, render both objects
            verts_cam_obj = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms, self.main_Ms_obj), 0)
            pred_rendered = self.cam_renderer.render_meshes([verts_cam, verts_cam_obj],
                                                            [pred['faces'], pred_obj['faces']])

        ## VISUALIZE ##
        rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
        depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
        seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy()).astype(np.uint8)

        gt_kpts2d = np.squeeze(self.gt_kpts2d.cpu().numpy())
        pred_kpts2d = np.squeeze(pred_kpts2d.cpu().detach().numpy())

        # check if gt kpts is nan (not detected)
        if np.isnan(gt_kpts2d).any():
            gt_kpts2d = np.zeros((21, 2))

        if flag_crop:
            # show cropped size of input (480, 640)
            rgb_input = np.squeeze(self.gt_rgb.cpu().numpy()).astype(np.uint8)
            depth_input = np.squeeze(self.gt_depth.cpu().numpy())
            seg_input = np.squeeze(self.gt_seg.cpu().numpy())

            # rendered image is original size (1080, 1920)
            rgb_mesh = rgb_mesh[self.bb[1]:self.bb[1]+self.bb[3], self.bb[0]:self.bb[0]+self.bb[2], :]
            depth_mesh = depth_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
            seg_mesh = seg_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]

            uv1 = np.concatenate((gt_kpts2d, np.ones_like(gt_kpts2d[:, :1])), 1)
            gt_kpts2d = (self.img2bb @ uv1.T).T
            uv1 = np.concatenate((pred_kpts2d, np.ones_like(pred_kpts2d[:, :1])), 1)
            pred_kpts2d = (self.img2bb @ uv1.T).T
        else:
            # show original size of input (1080, 1920)
            rgb_input, depth_input, seg_input, _, seg_obj = self.dataloaders[CFG_CAMID_SET.index(camID)].load_raw_image(frame)

        rgb_2d_gt = paint_kpts(None, rgb_mesh, gt_kpts2d)
        rgb_2d_pred = paint_kpts(None, rgb_mesh, pred_kpts2d)

        img_blend_gt = cv2.addWeighted(rgb_input, 0.5, rgb_2d_gt, 0.7, 0)
        img_blend_pred = cv2.addWeighted(rgb_input, 0.5, rgb_2d_pred, 0.7, 0)
        depth_gap = np.clip(np.abs(depth_input - depth_mesh), a_min=0.0, a_max=255.0).astype(np.uint8)
        seg_gap = ((seg_input - seg_mesh) * 255.0).astype(np.uint8)

        depth_gap *= seg_mesh
        seg_gap = seg_gap * seg_mesh * 255

        if not flag_crop:
            # resize images to (360, 640)
            img_blend_gt = cv2.resize(img_blend_gt, dsize=(640,360), interpolation=cv2.INTER_LINEAR)
            img_blend_pred = cv2.resize(img_blend_pred, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
            depth_gap = cv2.resize(depth_gap, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
            seg_gap = cv2.resize(seg_gap, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)

        blend_gt_name = "blend_gt_" + camID
        blend_pred_name = "blend_pred_" + camID
        blend_depth_name = "blend_depth_" + camID
        blend_seg_name = "blend_seg_" + camID

        cv2.imshow(blend_gt_name, img_blend_gt)
        cv2.imshow(blend_pred_name, img_blend_pred)
        cv2.imshow(blend_depth_name, depth_gap)
        # cv2.imshow(blend_seg_name, seg_gap)
        cv2.waitKey(1)

        # cv2.imwrite(os.path.join(save_path, 'img_blend_gt.png'), img_blend_gt)
        # cv2.imwrite(os.path.join(save_path, 'img_blend_pred.png'), img_blend_pred)


