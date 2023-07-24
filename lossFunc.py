import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.lossUtils import *
from utils.modelUtils import *

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