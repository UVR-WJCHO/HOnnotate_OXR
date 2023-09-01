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
        self.obj_scale = torch.FloatTensor([1.0, 1.0, 1.0]).to(self.device)

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
        self.gt_depth = torch.unsqueeze(torch.FloatTensor(gt_sample['depth']), 0).to(self.device)
        self.gt_seg = torch.unsqueeze(torch.FloatTensor(gt_sample['seg']), 0).to(self.device)

        if gt_sample['seg_obj'] != None:
            self.gt_seg_obj = torch.unsqueeze(torch.FloatTensor(gt_sample['seg_obj']), 0).to(self.device)
        else:
            self.gt_seg_obj = None

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

    def forward(self, pred, pred_obj, render, camIdx, frame, contact=False):
        # set gt data of current index & camera status
        self.set_gt(camIdx, frame)
        self.set_cam(camIdx)

        verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms, self.main_Ms), 0)
        joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms, self.main_Ms), 0)

        loss = {}
        if 'kpts2d' in self.loss_dict:
            if camIdx == 0 and frame == 26:
                loss['kpts2d'] = torch.zeros(1).to(self.device)
            else:
                pred_kpts2d = projectPoints(joints_cam, self.Ks)

                # loss_kpts2d = self.mse_loss(pred_kpts2d, self.gt_kpts2d.repeat(self.bs, 1, 1).to(self.device))
                loss_kpts2d = torch.sum(((pred_kpts2d - self.gt_kpts2d) ** 2).reshape(self.bs, -1), -1)
                loss['kpts2d'] = loss_kpts2d

                debug_pred = np.squeeze(pred_kpts2d.cpu().detach().numpy())
                debug_gt = np.squeeze(self.gt_kpts2d.cpu().detach().numpy())

        if 'reg' in self.loss_dict:
            pose_reg = self.compute_reg_loss(pred['pose'], self.pose_mean_tensor, self.pose_reg_tensor)
            shape_reg = torch.sum(
                ((pred['shape'] - torch.zeros_like(pred['shape'])) ** 2).view(self.bs, -1), -1)

            loss['reg'] = pose_reg + shape_reg

            if pred_obj is not None:
                pred_obj_rot = pred_obj['pose'].view(3, 4)[:, :-1]
                pred_obj_scale = torch.norm(pred_obj_rot, dim=0)
                loss_reg_obj = torch.abs(pred_obj_scale - self.obj_scale) * 100000.0
                loss['reg'] += torch.sum(loss_reg_obj)

        if render:
            pred_rendered = self.cam_renderer.render(verts_cam, pred['faces'])

            # TODO : need to combine both verts of hand/object
            if pred_obj is not None:
                verts_obj_cam = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms, self.main_Ms_obj), 0)
                pred_obj_rendered = self.cam_renderer.render(verts_obj_cam, pred_obj['faces'])

                # rgb_mesh_obj = np.squeeze((pred_obj_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                # cv2.imshow("rgb_mesh_obj", rgb_mesh_obj)
                # cv2.waitKey(1)

            if 'seg' in self.loss_dict:
                pred_seg = pred_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                pred_seg[pred_seg > 0] = 1.0

                seg_gap = torch.abs(pred_seg - self.gt_seg)
                seg_gap[pred_seg == 0] = 0.

                loss_seg = torch.sum(seg_gap.view(self.bs, -1), -1)
                # loss_seg = torch.clamp(loss_seg, min=0, max=5000)  # loss clipping following HOnnotate
                loss['seg'] = loss_seg

                # pred_seg = np.squeeze((pred_seg[0].cpu().detach().numpy()))
                # gt_seg = np.squeeze((self.gt_seg[0].cpu().detach().numpy()))
                # seg_gap = np.squeeze((seg_gap[0].cpu().detach().numpy()))

                # cv2.imshow("pred_seg", pred_seg)
                # cv2.imshow("gt_seg", gt_seg)
                # cv2.imshow("seg_gap", seg_gap)
                # cv2.waitKey(0)

                if pred_obj is not None:
                    pred_seg_obj = pred_obj_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0]+self.bb[2]]
                    pred_seg_obj[pred_seg_obj > 0] = 1
                    pred_seg_obj[pred_seg_obj < 0] = 0

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
                pred_depth = pred_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                depth_gap = torch.abs(pred_depth - self.gt_depth)
                depth_gap[self.gt_seg == 0] = 0
                # depth_gap[pred_depth == 0] = 0.
                depth_gap[depth_gap > 100] = 100.
                depth_gap /= 1000.0

                # pred_depth_vis = np.squeeze((pred_depth[0].cpu().detach().numpy())).astype(np.uint8)
                # gt_depth_vis = np.squeeze((self.gt_depth[0].cpu().detach().numpy())).astype(np.uint8)
                # depth_gap_vis = np.squeeze((depth_gap[0].cpu().detach().numpy()*255)).astype(np.uint8)
                # cv2.imshow("pred_depth", pred_depth_vis)
                # cv2.imshow("gt_depth_vis", gt_depth_vis)
                # cv2.imshow("depth_gap_vis", depth_gap_vis)
                # cv2.waitKey(1)

                loss_depth = torch.sum(depth_gap.view(self.bs, -1), -1)
                # loss_depth = torch.clamp(loss_depth, min=0, max=5000)  # loss clipping used in HOnnotate
                loss['depth'] = loss_depth

                if pred_obj is not None:
                    pred_depth_obj = pred_obj_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]

                    depth_obj_gap = torch.abs(pred_depth_obj - self.gt_depth)
                    depth_obj_gap[self.gt_seg_obj == 0] = 0

                    # depth_obj_gap[pred_depth_obj == 0] = 0.
                    depth_obj_gap[depth_obj_gap > 100] = 100.
                    depth_obj_gap /= 100.0

                    loss_depth_obj = torch.sum(depth_obj_gap.view(self.bs, -1), -1)
                    # loss_depth_obj = torch.clamp(loss_depth_obj, min=0, max=5000)  # loss clipping used in HOnnotate
                    loss['depth_obj'] = loss_depth_obj

                    # pred_depth_vis = np.squeeze((pred_depth_obj[0].cpu().detach().numpy())).astype(np.uint8)
                    # gt_depth_vis = np.squeeze((self.gt_depth[0].cpu().detach().numpy())).astype(np.uint8)
                    # depth_gap_vis = np.squeeze((depth_obj_gap[0].cpu().detach().numpy()*255)).astype(np.uint8)
                    # cv2.imshow("pred_depth", pred_depth_vis)
                    # cv2.imshow("gt_depth_vis", gt_depth_vis)
                    # cv2.imshow("depth_gap_vis", depth_gap_vis)
                    # cv2.waitKey(0)

        if 'contact' in self.loss_dict:
            if contact:
                hand_verts = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms, self.main_Ms), 0)
                obj_verts = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms, self.main_Ms_obj), 0)

                hand_pcd = Pointclouds(points=hand_verts)
                obj_mesh = Meshes(verts=obj_verts.detach(), faces=pred_obj['faces']) # optimize only hand meshes

                inter_dist = point_mesh_face_distance(obj_mesh, hand_pcd)
                contact_mask = inter_dist < CFG_CONTACT_THRESH

                loss['contact'] = inter_dist[contact_mask].sum()
            else:
                loss['contact'] = torch.tensor([0.0], requires_grad=True).cuda()



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
            rgb_input, depth_input, seg_input, seg_obj = self.dataloaders[CFG_CAMID_SET.index(camID)].load_raw_image(frame)

        rgb_2d_gt = paint_kpts(None, rgb_mesh, gt_kpts2d)
        rgb_2d_pred = paint_kpts(None, rgb_mesh, pred_kpts2d)
        rgb_seg = (rgb_input * seg_input[..., None]).astype(np.uint8)

        seg_mask = np.copy(seg_mesh)
        seg_mask[seg_mesh>0] = 1
        rgb_2d_pred *= seg_mask[..., None]

        img_blend_gt = cv2.addWeighted(rgb_input, 0.5, rgb_2d_gt, 0.7, 0)
        img_blend_pred = cv2.addWeighted(rgb_input, 1.0, rgb_2d_pred, 0.4, 0)
        img_blend_pred_seg = cv2.addWeighted(rgb_seg, 0.5, rgb_2d_pred, 0.7, 0)

        depth_gap = np.clip(np.abs(depth_input - depth_mesh), a_min=0.0, a_max=255.0).astype(np.uint8)
        seg_gap = ((seg_input - seg_mesh) * 255.0).astype(np.uint8)
        depth_gap *= seg_mask
        seg_gap *= seg_mask * 255

        if not flag_crop:
            # resize images to (360, 640)
            img_blend_gt = cv2.resize(img_blend_gt, dsize=(640,360), interpolation=cv2.INTER_LINEAR)
            img_blend_pred = cv2.resize(img_blend_pred, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
            img_blend_pred_seg = cv2.resize(img_blend_pred_seg, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
            depth_gap = cv2.resize(depth_gap, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
            seg_gap = cv2.resize(seg_gap, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)

        blend_gt_name = "blend_gt_" + camID + "_" + str(frame)
        blend_pred_name = "blend_pred_" + camID + "_" + str(frame)
        blend_pred_seg_name = "blend_pred_seg_" + camID + "_" + str(frame)
        blend_depth_name = "blend_depth_" + camID + "_" + str(frame)
        blend_seg_name = "blend_seg_" + camID + "_" + str(frame)

        # try:
        #     cv2.imshow(blend_gt_name, img_blend_gt)
        #     cv2.imshow(blend_pred_name, img_blend_pred)
        #     # cv2.imshow(blend_pred_seg_name, img_blend_pred_seg)
        #     # cv2.imshow(blend_depth_name, depth_gap)
        #     # cv2.imshow(blend_seg_name, seg_gap)
        #     cv2.waitKey(0)
        # except:
        #     print("headless server")

        cv2.imwrite(os.path.join(save_path, blend_pred_name + '.png'), img_blend_pred)
        # cv2.imwrite(os.path.join(save_path, blend_pred_seg_name + '.png'), img_blend_pred_seg)
        cv2.imwrite(os.path.join(save_path, blend_depth_name + '.png'), depth_gap)

        # save meshes
        import trimesh
        hand_verts = mano3DToCam3D(pred['verts'], self.Ms, self.main_Ms)
        hand = trimesh.Trimesh(hand_verts.detach().cpu().numpy(), pred['faces'][0].detach().cpu().numpy())
        hand.export(os.path.join(save_path, f'mesh_hand_{camID}_{frame}.obj'))

        obj_verts = mano3DToCam3D(pred_obj['verts'], self.Ms, self.main_Ms_obj)
        obj = trimesh.Trimesh(obj_verts.detach().cpu().numpy(), pred_obj['faces'][0].detach().cpu().numpy())
        obj.export(os.path.join(save_path, f'mesh_obj_{camID}_{frame}.obj'))
