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
    def __init__(self, device='cpu', bs=1):
        super(MultiViewLossFunc, self).__init__()
        self.device = device
        self.bs = bs
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

        self.pose_reg_tensor, self.pose_mean_tensor = self.get_pose_constraint_tensor()

    def get_pose_constraint_tensor(self):
        pose_mean_tensor = torch.tensor(params.pose_mean_list).cuda()
        pose_reg_tensor = torch.tensor(params.pose_reg_list).cuda()
        return pose_reg_tensor, pose_mean_tensor

    def compute_reg_loss(self, mano_tensor, pose_mean_tensor, pose_reg_tensor):
        reg_loss = ((mano_tensor - pose_mean_tensor) ** 2) * pose_reg_tensor
        return reg_loss

    def set_gt(self, gt_sample, cam_params, cam_renderer, loss_dict, main_cam_params):
        self.bb = np.asarray(gt_sample['bb']).astype(int)
        self.img2bb = gt_sample['img2bb']
        self.gt_kpts2d = torch.FloatTensor(gt_sample['kpts2d'])
        self.gt_kpts3d = torch.FloatTensor(gt_sample['kpts3d'])

        self.gt_rgb = torch.FloatTensor(gt_sample['rgb']).to(self.device)
        self.gt_depth = torch.FloatTensor(gt_sample['depth']).to(self.device)
        self.gt_seg = torch.FloatTensor(gt_sample['seg']).to(self.device)

        Ks, Ms, _ = cam_params
        self.Ks = torch.FloatTensor(Ks).to(self.device)
        self.Ms = torch.FloatTensor(Ms).to(self.device)
        self.cam_renderer = cam_renderer

        self.loss_dict = loss_dict

        main_Ks, main_Ms, _ = main_cam_params
        self.main_Ks = torch.FloatTensor(main_Ks).to(self.device)
        self.main_Ms = torch.FloatTensor(main_Ms).to(self.device)

    def forward(self, pred, render=False):
        verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms, self.main_Ms), 0)
        joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms, self.main_Ms), 0)


        loss = {}
        if 'kpts2d' in self.loss_dict:
            pred_kpts2d = projectPoints(joints_cam, self.Ks)
            self.pred_kpts2d = pred_kpts2d.clone().detach()

            # loss_kpts2d = self.mse_loss(pred_kpts2d, self.gt_kpts2d.repeat(self.bs, 1, 1).to(self.device))
            loss_kpts2d = torch.sum((pred_kpts2d - self.gt_kpts2d.repeat(self.bs, 1, 1).to(self.device)) ** 2)
            loss['kpts2d'] = loss_kpts2d

        if 'reg' in self.loss_dict:
            pose_reg = self.compute_reg_loss(pred['pose'], self.pose_mean_tensor, self.pose_reg_tensor)
            shape_reg = torch.sum(
                ((pred['shape'] - torch.zeros_like(pred['shape'])) ** 2).view(self.bs, -1), -1)

            loss['reg'] = pose_reg + shape_reg

        if render:
            self.pred_rendered = self.cam_renderer.render(verts_cam, pred['faces'])
            if 'seg' in self.loss_dict:
                pred_seg = self.pred_rendered['seg'][0, self.bb[1]:self.bb[1] + self.bb[3],
                           self.bb[0]:self.bb[0] + self.bb[2], 0]

                loss_seg = self.mse_loss(pred_seg, self.gt_seg)
                loss_seg = torch.clamp(loss_seg, min=0, max=2000)  # loss clipping following HOnnotate
                loss['seg'] = loss_seg

            if 'depth' in self.loss_dict:
                pred_depth = self.pred_rendered['depth'][0, self.bb[1]:self.bb[1] + self.bb[3],
                           self.bb[0]:self.bb[0] + self.bb[2], 0]
                pred_depth[pred_depth == -1] = 0.

                # loss_depth = torch.sum(((depth_rendered - self.depth_ref / self.scale) ** 2).view(self.batch_size, -1),
                #                        -1) * 0.00012498664727900177  # depth scale used in HOnnotate
                loss_depth = self.mse_loss(pred_depth, self.gt_depth)
                loss_depth = torch.clamp(loss_depth, min=0, max=2000)  # loss clipping used in HOnnotate
                loss['depth'] = loss_depth


        return loss

    def visualize(self, save_path, camID):
        # input is cropped size (480, 640)
        rgb_input = np.squeeze(self.gt_rgb.cpu().numpy()).astype(np.uint8)
        depth_input = np.squeeze(self.gt_depth.cpu().numpy())
        seg_input = np.squeeze(self.gt_seg.cpu().numpy())

        # rendered image is original size (1080, 1920)
        rgb_mesh = np.squeeze((self.pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
        depth_mesh = np.squeeze(self.pred_rendered['depth'][0].cpu().detach().numpy())
        seg_mesh = np.squeeze(self.pred_rendered['seg'][0].cpu().detach().numpy()).astype(np.uint8)

        rgb_mesh = rgb_mesh[self.bb[1]:self.bb[1]+self.bb[3], self.bb[0]:self.bb[0]+self.bb[2], :]
        depth_mesh = depth_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
        seg_mesh = seg_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]


        gt_kpts2d = np.squeeze(self.gt_kpts2d.cpu().numpy())
        pred_kpts2d = np.squeeze(self.pred_kpts2d.cpu().numpy())

        uv1 = np.concatenate((gt_kpts2d, np.ones_like(gt_kpts2d[:, :1])), 1)
        gt_kpts2d_bb = (self.img2bb @ uv1.T).T
        rgb_2d_gt = paint_kpts(None, rgb_mesh, gt_kpts2d_bb)

        uv1 = np.concatenate((pred_kpts2d, np.ones_like(pred_kpts2d[:, :1])), 1)
        pred_kpts2d_bb = (self.img2bb @ uv1.T).T
        rgb_2d_pred = paint_kpts(None, rgb_mesh, pred_kpts2d_bb)

        img_blend_gt = cv2.addWeighted(rgb_input, 0.5, rgb_2d_gt, 0.7, 0)
        img_blend_pred = cv2.addWeighted(rgb_input, 0.5, rgb_2d_pred, 0.7, 0)

        blend_gt_name = "img_blend_gt_" + camID
        blend_pred_name = "img_blend_pred_" + camID

        cv2.imshow(blend_gt_name, img_blend_gt)
        cv2.imshow(blend_pred_name, img_blend_pred)
        cv2.waitKey(1)

        # cv2.imwrite(os.path.join(save_path, 'img_blend_gt.png'), img_blend_gt)
        # cv2.imwrite(os.path.join(save_path, 'img_blend_pred.png'), img_blend_pred)


class SingleViewLossFunc(nn.Module):
    def __init__(self, device='cpu', bs=1):
        super(SingleViewLossFunc, self).__init__()
        self.device = device
        self.bs = bs
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def set_gt(self, gt_sample, cam_params, cam_renderer, loss_dict):
        self.bb = gt_sample['bb']
        self.img2bb = gt_sample['img2bb']
        self.gt_kpts2d = torch.FloatTensor(gt_sample['kpts2d'])
        self.gt_seg = gt_sample['seg']
        self.gt_rgb = gt_sample['rgb']
        self.Ks = torch.FloatTensor(cam_params['Ks'])
        self.Ms = torch.FloatTensor(cam_params['Ms'])
        self.cam_renderer = cam_renderer

        self.loss_dict = loss_dict
        
    def forward(self, pred):
        verts_world = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms, self.Ms), 0)
        joints_world = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms, self.Ms), 0)
        self.pred_render_results = self.cam_renderer.render(verts_world, pred['faces'])
        loss = {}
        if 'kpts2d' in self.loss_dict:
            pred_kpts2d = projectPoints(joints_world, self.Ks)
            # pred_kpts2d[torch.isnan(pred_kpts2d)] = 0.0
            self.pred_kpts2d = pred_kpts2d
            # loss_kpts2d = self.mse_loss(pred_kpts2d, self.gt_kpts2d.repeat(self.bs, 1, 1).to(self.device))
            loss_kpts2d = torch.sum((pred_kpts2d - self.gt_kpts2d.repeat(self.bs, 1, 1).to(self.device)) ** 2 )
            loss['kpts2d'] = loss_kpts2d

        if 'seg' in self.loss_dict:
            pred_seg = self.pred_render_results['seg']
            loss_seg = self.mse_loss(pred_seg, self.gt_seg)
            loss['seg'] = loss_seg

        if 'reg' in self.loss_dict:
            pose_reg, pose_mean = get_pose_constraint_tensor()
            loss_reg = self.mse_loss(pred['pose'], pose_mean) * pose_reg
            loss_reg = torch.sum(((pred['pose'] - pose_mean) ** 2 ) * pose_reg) + torch.sum((pred['shape'] - torch.zeros_like(pred['shape'])) ** 2 )
            loss['reg'] = torch.sum(loss_reg)

        return loss
    
    def visualize(self, save_path):
        """
        TODO
        """
        cv2.imwrite(os.path.join(save_path, 'gt_seg.png'), self.pred_render_results['image'][0].cpu().detach().numpy()*255)
        img_1 = (self.pred_render_results['image'][0].cpu().detach().numpy() * 255.0).astype(np.uint8)
        img_1 = img_1[int(self.bb[1]):int(self.bb[1])+self.bb[3], int(self.bb[0]):int(self.bb[0])+self.bb[2], :]
        img_2 = self.gt_rgb
        img_3 = cv2.addWeighted(img_1, 0.5, img_2, 0.7, 0)
        cv2.imwrite(os.path.join(save_path, 'debug.png'), img_3)

        kpts2d_homo = torch.concat((self.pred_kpts2d, torch.ones_like(self.pred_kpts2d[:, :, :1])), 2)
        uv = (self.img2bb @ kpts2d_homo[0].cpu().detach().numpy().T).T
        img_4 = paint_kpts(None, img_3, uv)
        cv2.imwrite(os.path.join(save_path, 'debug2.png'), img_4)

