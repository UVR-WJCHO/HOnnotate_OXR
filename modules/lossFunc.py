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
import time
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops import SubdivideMeshes

import csv
import trimesh
import open3d as o3d
import pandas as pd
from modules.utils.processing import gen_trans_from_patch_cv


class MultiViewLossFunc(nn.Module):
    def __init__(self, device='cuda', bs=1, dataloaders=None, renderers=None, losses=None):
        super(MultiViewLossFunc, self).__init__()
        self.device = device
        self.bs = bs
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.pose_reg_tensor, self.pose_mean_tensor = self.get_pose_constraint_tensor()
        self.dataloaders = dataloaders
        self.cam_renderer = renderers
        self.loss_dict = losses
        self.obj_scale = torch.FloatTensor([1.0, 1.0, 1.0]).to(self.device)
        self.h = torch.tensor([[0, 0, 0, 1]]).to(device)
        self.Ks = []
        self.Ms = []
        for camIdx in range(len(CFG_CAMID_SET)):
            cam_params = self.dataloaders[camIdx].cam_parameter
            Ks, Ms, _ = cam_params
            self.Ks.append(Ks)
            self.Ms.append(Ms)

        self.default_zero = torch.tensor([0.0], requires_grad=True).to(self.device)

        # self.vis = self.set_visibility_weight(CFG_CAM_PER_FINGER_VIS)

        self.const = Constraints()
        self.rot_min = torch.tensor(np.asarray(params.rot_min_list)).to(self.device)
        self.rot_max = torch.tensor(np.asarray(params.rot_max_list)).to(self.device)

        self.prev_hand_pose = None
        self.prev_hand_shape = None

        self.temp_weight = CFG_temporal_loss_weight

        self.gt_obj_marker = None
        self.vertIDpermarker = None

        self.obj_mesh_dense = None

        self.init_bbox = {}
        self.init_bbox["mas"] = [400, 60, 1120, 960]
        self.init_bbox["sub1"] = [360, 0, 1120, 960]
        self.init_bbox["sub2"] = [640, 180, 640, 720]
        self.init_bbox["sub3"] = [680, 180, 960, 720]

    def reset_prev_pose(self):
        self.prev_hand_pose = None
        self.prev_hand_shape = None

    def set_visibility_weight(self, CFG_W):
        vis = {}
        for camIdx in range(len(CFG_CAMID_SET)):
            vis_cam = np.ones(21)
            weights = CFG_W[CFG_CAMID_SET[camIdx]]
            # for weight in weights:
            vis_cam[1:5] = weights[0]
            vis_cam[5:9] = weights[1]
            vis_cam[9:13] = weights[2]
            vis_cam[13:17] = weights[3]
            vis_cam[17:21] = weights[4]

            vis_cam = torch.unsqueeze(torch.FloatTensor(vis_cam).to(self.device), 1)
            vis[camIdx] = vis_cam
        return vis


    def set_object_main_extrinsic(self, obj_main_cam_idx):
        self.main_Ms_obj = self.Ms[obj_main_cam_idx]

    def set_object_marker_pose(self, obj_marker_cam_pose, marker_valid_idx, obj_class, CFG_DATE, grasp_idx):
        self.vertIDpermarker = CFG_vertspermarker[str(CFG_DATE)][str(obj_class)]

        if int(obj_class.split('_')[0]) == 29:
            if grasp_idx == 12:
                self.vertIDpermarker = self.vertIDpermarker[0]
            else:
                self.vertIDpermarker = self.vertIDpermarker[1]

        self.marker_valid_idx = marker_valid_idx
        if obj_marker_cam_pose != None:
            self.gt_obj_marker = torch.unsqueeze(obj_marker_cam_pose, 0)
        else:
            self.gt_obj_marker = None


    def get_pose_constraint_tensor(self):
        pose_mean_tensor = torch.tensor(params.pose_mean_list).cuda()
        pose_reg_tensor = torch.tensor(params.pose_reg_list).cuda()
        return pose_reg_tensor, pose_mean_tensor

    def compute_reg_loss(self, mano_tensor, pose_mean_tensor, pose_reg_tensor):
        reg_loss = ((mano_tensor - pose_mean_tensor) ** 2) * pose_reg_tensor
        return torch.sum(reg_loss, -1)


    def set_gt_nobb(self, camIdx, camID, frame):
        gt_sample = self.dataloaders[camIdx][frame]
        self.bb = self.init_bbox[str(camID)]
        self.gt_rgb = gt_sample['rgb_raw'][self.bb[1]:self.bb[1]+self.bb[3], self.bb[0]:self.bb[0]+self.bb[2], :]
        self.gt_depth = gt_sample['depth_raw'][0, self.bb[1]:self.bb[1]+self.bb[3], self.bb[0]:self.bb[0]+self.bb[2]]

        bb_c_x = float(self.bb[0] + 0.5 * self.bb[2])
        bb_c_y = float(self.bb[1] + 0.5 * self.bb[3])
        bb_width = float(self.bb[2])
        bb_height = float(self.bb[3])
        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, 640, 480, 1.0, 0.0, (0.0, 0.0))
        self.img2bb = trans

    def set_gt(self, camIdx, frame):
        gt_sample = self.dataloaders[camIdx][frame]

        self.bb = gt_sample['bb']
        self.img2bb = gt_sample['img2bb']
        self.gt_kpts2d = gt_sample['kpts2d']
        self.gt_kpts3d = gt_sample['kpts3d']

        self.gt_rgb = gt_sample['rgb']
        self.gt_depth = gt_sample['depth']
        self.gt_depth_obj = gt_sample['depth_obj']

        # if no gt seg, mask is 1 in every pixel
        self.gt_seg = gt_sample['seg']
        self.gt_seg_obj = gt_sample['seg_obj']

        self.gt_visibility = gt_sample['visibility']

        if gt_sample['tip2d'] is not None:
            self.gt_tip2d = gt_sample['tip2d']
            self.valid_tip_idx = gt_sample['validtip']
        else:
            self.gt_tip2d = None


    def set_main_cam(self, main_cam_idx=0):
        # main_cam_params = self.dataloaders[main_cam_idx].cam_parameter

        self.main_Ks = self.Ks[main_cam_idx]
        self.main_Ms = self.Ms[main_cam_idx]

    def forward(self, pred, pred_obj, camIdxSet, frame, loss_dict, contact=False, penetration=False, parts=-1, flag_headless=False):

        self.loss_dict = loss_dict

        render = False
        if 'depth' in loss_dict or 'seg' in loss_dict:
            render = True

        verts_set = {}
        verts_obj_set = {}
        joints_set = {}
        pred_render_set = {}
        pred_obj_render_set = {}

        ## compute per cam predictions
        for camIdx in camIdxSet:
            verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms[camIdx]), 0)
            joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms[camIdx]), 0)
            verts_set[camIdx] = verts_cam
            joints_set[camIdx] = joints_cam

            if render:
                pred_rendered = self.cam_renderer[camIdx].render(verts_cam, pred['faces'])
                pred_render_set[camIdx] = pred_rendered

                if pred_obj is not None:
                    verts_obj_cam = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms[camIdx]), 0)
                    pred_obj_rendered = self.cam_renderer[camIdx].render(verts_obj_cam, pred_obj['faces'])

                    verts_obj_set[camIdx] = verts_obj_cam
                    pred_obj_render_set[camIdx] = pred_obj_rendered

        ## compute losses per cam
        losses_cam = {}
        for camIdx in camIdxSet:
            loss = {}

            self.set_gt(camIdx, frame)

            pred_kpts2d = projectPoints(joints_set[camIdx], self.Ks[camIdx])

            if 'kpts_tip' in self.loss_dict:
                if self.gt_tip2d != None:
                    pred_kpts2d_tip = pred_kpts2d[:, self.valid_tip_idx, :]
                    loss_tip = (pred_kpts2d_tip - self.gt_tip2d) ** 2
                    loss_tip = torch.sum(loss_tip.reshape(self.bs, -1), -1)
                    loss['kpts_tip'] = loss_tip * 0.5e2
                else:
                    loss['kpts_tip'] = self.default_zero

            if 'kpts_palm' in self.loss_dict:
                pred_kpts2d_palm = pred_kpts2d[:, CFG_PALM_IDX, :]
                gt_kpts2d_palm = self.gt_kpts2d[:, CFG_PALM_IDX, :]

                loss_palm = (pred_kpts2d_palm - gt_kpts2d_palm) ** 2
                loss_palm = torch.sum(loss_palm.reshape(self.bs, -1), -1)
                loss['kpts_palm'] = loss_palm

            if 'kpts2d' in self.loss_dict:
                # all
                if parts == -1 or parts == 2:
                    loss_kpts2d = ((pred_kpts2d - self.gt_kpts2d) ** 2) * self.gt_visibility
                # parts
                else:
                    valid_idx = CFG_valid_index[parts]
                    loss_kpts2d = ((pred_kpts2d[:, valid_idx, :] - self.gt_kpts2d[:, valid_idx, :]) ** 2) \
                                  * self.gt_visibility[valid_idx]

                loss_kpts2d = torch.sum(loss_kpts2d.reshape(self.bs, -1), -1)
                loss['kpts2d'] = loss_kpts2d * 5e0

            if 'depth_rel' in self.loss_dict:
                joint_depth_rel = joints_set[camIdx][:, :, -1] - joints_set[camIdx][:, 0, -1]
                gt_depth_rel = self.gt_kpts3d[:, :, -1] - self.gt_kpts3d[:, 0, -1]

                joint_scale = joint_depth_rel[:, 1] - joint_depth_rel[:, 0]
                gt_scale = gt_depth_rel[:, 1] - gt_depth_rel[:, 0]
                ratio = joint_scale / gt_scale

                gt_depth_rel = torch.mul(gt_depth_rel, ratio)
                if parts == -1 or parts == 2:
                    loss_depth_rel = self.mse_loss(joint_depth_rel, gt_depth_rel)
                else:
                    valid_idx = CFG_valid_index[parts]
                    loss_depth_rel = self.mse_loss(joint_depth_rel[:, valid_idx], gt_depth_rel[:, valid_idx])
                loss['depth_rel'] = loss_depth_rel * 5e1

            if 'pose_obj' in self.loss_dict:
                pred_obj_verts_marker = pred_obj['verts'][:, self.vertIDpermarker, :] * 10.0
                gt_obj_verts_marker = self.gt_obj_marker

                pred_obj_verts_marker = pred_obj_verts_marker[:, self.marker_valid_idx, :]

                if gt_obj_verts_marker is not None:
                    loss_pose_obj = (pred_obj_verts_marker - gt_obj_verts_marker) ** 2
                    loss_pose_obj = torch.sum(loss_pose_obj.reshape(self.bs, -1), -1)
                    loss['pose_obj'] = loss_pose_obj * 1e0
                else:
                    loss['pose_obj'] = self.default_zero

            if render:
                pred_rendered = pred_render_set[camIdx]
                if pred_obj is not None:
                    pred_obj_rendered = pred_obj_render_set[camIdx]

                    # rgb_mesh_obj = np.squeeze((pred_obj_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                    # cv2.imshow("rgb_mesh_obj", rgb_mesh_obj)
                    # cv2.waitKey(1)

                if 'seg' in self.loss_dict:
                    pred_seg = pred_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                    pred_seg = torch.div(pred_seg, torch.max(pred_seg))
                    pred_seg = torch.ceil(pred_seg)
                    # pred_seg[pred_seg==0] = 10.

                    # a = self.gt_seg.clone().cpu().detach().numpy()
                    # b = pred_seg.clone().cpu().detach().numpy()

                    # seg_gap = torch.abs(pred_seg - self.gt_seg)
                    # loss_seg = torch.sum(seg_gap.view(self.bs, -1), -1)
                    self.cam_renderer[camIdx].register_seg(self.gt_seg)
                    loss_seg, seg_gap = self.cam_renderer[camIdx].compute_seg_loss(pred_seg)
                    loss['seg'] = loss_seg * 1e1

                    # if camIdx == 3:
                    #     pred_seg = np.squeeze((pred_seg[0].cpu().detach().numpy()))
                    #     gt_seg = np.squeeze((self.gt_seg[0].cpu().detach().numpy()))
                    #     seg_gap = np.squeeze((seg_gap[0].cpu().detach().numpy()))
                    #     cv2.imshow("pred_seg"+str(camIdx), pred_seg)
                    #     cv2.imshow("gt_seg"+str(camIdx), gt_seg)
                    #     cv2.imshow("seg_gap"+str(camIdx), seg_gap)
                    #     cv2.waitKey(0)

                    if pred_obj is not None:
                        pred_seg_obj = pred_obj_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0]+self.bb[2]]
                        pred_seg_obj = torch.div(pred_seg_obj, torch.max(pred_seg_obj))
                        pred_seg_obj = torch.ceil(pred_seg_obj)
                        # pred_seg_obj[pred_seg_obj == 0] = 10.

                        a = self.gt_seg_obj.clone().cpu().detach().numpy()
                        b = pred_seg_obj.clone().cpu().detach().numpy()

                        self.cam_renderer[camIdx].register_seg(self.gt_seg_obj)
                        loss_seg_obj, seg_obj_gap = self.cam_renderer[camIdx].compute_seg_loss(pred_seg_obj)

                        loss['seg_obj'] = loss_seg_obj * 0.5e1

                        # if camIdx == 0:
                        #     pred_seg_obj = np.squeeze((pred_seg_obj[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                        #     seg_obj_gap = np.squeeze((seg_obj_gap[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                        #     gt_seg_obj = np.squeeze((self.gt_seg_obj[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                        #
                        #     if not flag_headless:
                        #         cv2.imshow("pred_seg_obj", pred_seg_obj)
                        #         cv2.imshow("gt_seg_obj", gt_seg_obj)
                        #         cv2.imshow("seg_obj_gap", seg_obj_gap)
                        #         cv2.waitKey(1)
                        #     else:
                        #         cv2.imwrite(os.path.join("./for_headless_server", 'pred_seg_obj.png'), pred_seg_obj)
                        #         cv2.imwrite(os.path.join("./for_headless_server", 'gt_seg_obj.png'), gt_seg_obj)
                        #         cv2.imwrite(os.path.join("./for_headless_server", 'seg_obj_gap.png'), seg_obj_gap)

                if 'depth' in self.loss_dict:
                    pred_depth = pred_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3],
                                 self.bb[0]:self.bb[0] + self.bb[2]]

                    a = self.gt_depth.clone().cpu().detach().numpy()
                    b = pred_depth.clone().cpu().detach().numpy()

                    self.cam_renderer[camIdx].register_depth(self.gt_depth)
                    loss_depth, depth_gap = self.cam_renderer[camIdx].compute_depth_loss(pred_depth)

                    # depth_gap = torch.abs(pred_depth - self.gt_depth)
                    # depth_gap[pred_depth == 0] = 0

                    # loss_depth = torch.mean(depth_gap.view(self.bs, -1), -1)
                    loss['depth'] = loss_depth * 1e0

                    # if camIdx == 0:
                    #     pred_depth_vis = np.squeeze((pred_depth[0].cpu().detach().numpy())* 25).astype(np.uint8)
                    #     gt_depth_vis = np.squeeze((self.gt_depth[0].cpu().detach().numpy())* 25).astype(np.uint8)
                    #     depth_gap_vis = np.squeeze((depth_gap[0].cpu().detach().numpy())).astype(np.uint8)
                    #     cv2.imshow("pred_depth"+str(camIdx), pred_depth_vis)
                    #     cv2.imshow("gt_depth_vis"+str(camIdx), gt_depth_vis)
                    #     cv2.imshow("depth_gap_vis"+str(camIdx), depth_gap_vis)
                    #     cv2.waitKey(0)

                    if pred_obj is not None:
                        pred_depth_obj = pred_obj_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3],
                                         self.bb[0]:self.bb[0] + self.bb[2]]

                        a = self.gt_depth_obj.clone().cpu().detach().numpy()
                        b = pred_depth_obj.clone().cpu().detach().numpy()

                        self.cam_renderer[camIdx].register_depth(self.gt_depth_obj)
                        loss_depth_obj, depth_obj_gap = self.cam_renderer[camIdx].compute_depth_loss(pred_depth_obj)

                        # depth_obj_gap = torch.abs(pred_depth_obj - self.gt_depth_obj)
                        # depth_obj_gap[self.gt_depth_obj == 0] = 0
                        # loss_depth_obj = torch.mean(depth_obj_gap.view(self.bs, -1), -1)

                        loss['depth_obj'] = loss_depth_obj * 1e-1
                        # pred_depth_vis = np.squeeze((pred_depth_obj[0].cpu().detach().numpy()*25)).astype(np.uint8)
                        # gt_depth_vis = np.squeeze((self.gt_depth_obj[0].cpu().detach().numpy()*25)).astype(np.uint8)
                        # depth_gap_vis = np.squeeze((depth_obj_gap[0].cpu().detach().numpy())).astype(np.uint8)
                        # cv2.imshow("pred_depth"+str(camIdx), pred_depth_vis)
                        # cv2.imshow("gt_depth_vis"+str(camIdx), gt_depth_vis)
                        # cv2.imshow("depth_gap_vis"+str(camIdx), depth_gap_vis)
                        # cv2.waitKey(1)
                        # if camIdx == 0:
                        #     pred_depth_vis = np.squeeze((pred_depth_obj[0].cpu().detach().numpy())/10.0).astype(np.uint8)
                        #     gt_depth_vis = np.squeeze((self.gt_depth[0].cpu().detach().numpy())/10.0).astype(np.uint8)
                        #     depth_gap_vis = np.squeeze((depth_obj_gap[0].cpu().detach().numpy())).astype(np.uint8)
                        #     if not flag_headless:
                        #         cv2.imshow("pred_depth", pred_depth_vis)
                        #         cv2.imshow("gt_depth_vis", gt_depth_vis)
                        #         cv2.imshow("depth_gap_vis", depth_gap_vis)
                        #         cv2.waitKey(1)
                        #     else:
                        #         cv2.imwrite(os.path.join("./for_headless_server", 'obj_pred_depth.png'), pred_depth_vis)
                        #         cv2.imwrite(os.path.join("./for_headless_server", 'obj_gt_depth.png'), gt_depth_vis)
                        #         cv2.imwrite(os.path.join("./for_headless_server", 'obj_depth_gap.png'), depth_gap_vis)

            losses_cam[camIdx] = loss

        ## compute single losses
        losses_single = {}
        if 'reg' in self.loss_dict:
            pose_reg = torch.sum((pred['pose'] ** 2).view(self.bs, -1), -1)
            shape_reg = torch.sum((pred['shape'] ** 2).view(self.bs, -1), -1)

            ## wrong adoption? check parameter's order
            thetaConstMin, thetaConstMax = self.const.getHandJointConstraints(pred['pose'])
            phyConst = torch.sum(thetaConstMin ** 2 + thetaConstMax ** 2)

            losses_single['reg'] = pose_reg * 1e2 + shape_reg * 1e2 + phyConst * 1e4

            # if pred_obj is not None:
            #     #pred_obj_rot = pred_obj['pose'].view(3, 4)[:, :-1]
            #     #pred_obj_scale = torch.norm(pred_obj_rot, dim=0)
            #     #loss_reg_obj = torch.abs(pred_obj_scale - self.obj_scale) * 1e4
            #     pred_obj_pose_diff = torch.norm(pred_obj['pose'][:, :-3], dim=0)
            #     losses_single['reg'] += torch.sum(pred_obj_pose_diff)

        if contact or penetration:
            hand_pcd = Pointclouds(points=verts_set[camIdx])
            obj_mesh = Meshes(verts=verts_obj_set[camIdx].detach(),
                              faces=pred_obj['faces'])  # optimize only hand meshes

            if 'contact' in self.loss_dict:
                if contact and pred_obj is not None:
                    inter_dist = point_mesh_face_distance(obj_mesh, hand_pcd)

                    # debug = inter_dist.clone().cpu().detach().numpy()
                    contact_mask = inter_dist < CFG_CONTACT_DIST

                    loss['contact'] = inter_dist[contact_mask].sum() * 1E-1
                    pred['contact'] = torch.ones(778).cuda() * -1.
                    pred['contact'][contact_mask] = inter_dist[contact_mask]

                else:
                    loss['contact'] = self.default_zero
                    pred['contact'] = torch.ones(778).cuda() * -1.

            if 'penetration' in self.loss_dict:
                if penetration and pred_obj is not None:
                    if self.obj_mesh_dense is None:
                        for _ in range(5):
                            obj_mesh = SubdivideMeshes()(obj_mesh)
                            self.obj_mesh_dense = obj_mesh
                    else:
                        obj_mesh = self.obj_mesh_dense
                    verts_obj_norm = obj_mesh.verts_normals_padded()
                    collide_ids_hand, collide_ids_obj = collision_check(verts_obj_set[camIdx], verts_obj_norm, verts_set[camIdx], chamferDist())

                    if collide_ids_hand is not None:
                        loss['penetration'] = (verts_obj_set[camIdx][:, collide_ids_obj] - verts_set[camIdx][:, collide_ids_hand]).square().mean() * 1e2
                    else:
                        loss['penetration'] = self.default_zero


        if 'temporal' in self.loss_dict:
            if self.prev_hand_pose == None:
                losses_single['temporal'] = self.default_zero
            else:
                loss_temporal = torch.sum((pred['pose'] - self.prev_hand_pose) ** 2) + \
                                   torch.sum((pred['shape'] - self.prev_hand_shape) ** 2)
                losses_single['temporal'] = loss_temporal * self.temp_weight

        if pred_obj is None:
            return losses_cam, losses_single
        else:
            return losses_cam, losses_single, pred['contact']
    
    def set_for_evaluation(self):
        #=====Set evaluation metrics dataframe=====#
        self.dfs = {}
        self.dfs['kpts_precision'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['kpts_recall'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['kpts_f1'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['mesh_precision'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['mesh_recall'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['mesh_f1'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['depth_precision'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['depth_recall'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        self.dfs['depth_f1'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])

        self.total_metrics = [0.] * 9

        for df in self.dfs.values():
            df.index.name = "frame"
    
    def evaluation(self, pred, pred_obj, camIdxSet, frame):
        print("Evaluation")
        kpts_precision = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}
        kpts_recall = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}
        kpts_f1 = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}

        mesh_precision = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}
        mesh_recall = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}
        mesh_f1 = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}

        depth_precision = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}
        depth_recall = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}
        depth_f1 = {"mas":0., "sub1":0., "sub2":0., "sub3":0.}

        for camIdx in camIdxSet:
            camID = CFG_CAMID_SET[camIdx]
            self.set_gt(camIdx, frame)
            joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms[camIdx]), 0)
            verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms[camIdx]), 0)
            verts_cam_obj = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms[camIdx]), 0)

            pred_rendered = self.cam_renderer[camIdx].render_meshes([verts_cam, verts_cam_obj],
                                                            [pred['faces'], pred_obj['faces']], flag_rgb=True)
            pred_rendered_hand_only = self.cam_renderer[camIdx].render(verts_cam,pred['faces'], flag_rgb=True)
            pred_rendered_obj_only = self.cam_renderer[camIdx].render(verts_cam_obj, pred_obj['faces'], flag_rgb=True)
            
            depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
            seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy()).astype(np.uint8)

            hand_depth = np.squeeze(pred_rendered_hand_only['depth'][0].cpu().detach().numpy())
            obj_depth = np.squeeze(pred_rendered_obj_only['depth'][0].cpu().detach().numpy())
            hand_depth[hand_depth == 10] = 0
            hand_depth *= 1000.
            obj_depth[obj_depth == 10] = 0
            obj_depth *= 1000.

            obj_seg_masked = np.copy(obj_depth)
            hand_seg_masked = np.where(abs(depth_mesh - obj_depth) < 1.0, 0, 1)
            obj_seg_masked = np.where(abs(depth_mesh - hand_depth) < 1.0, 0, 1)

            pred_kpts2d = projectPoints(joints_cam, self.Ks[camIdx])
            pred_kpts2d = np.squeeze(pred_kpts2d.clone().cpu().detach().numpy())
            gt_kpts2d = np.squeeze(self.gt_kpts2d.clone().cpu().numpy())

            hand_seg_masked = hand_seg_masked[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
            obj_seg_masked = obj_seg_masked[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
            hand_depth = hand_depth[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
            obj_depth = obj_depth[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]

            uv1 = np.concatenate((gt_kpts2d, np.ones_like(gt_kpts2d[:, :1])), 1)
            gt_kpts2d = (self.img2bb @ uv1.T).T
            uv1 = np.concatenate((pred_kpts2d, np.ones_like(pred_kpts2d[:, :1])), 1)
            pred_kpts2d = (self.img2bb @ uv1.T).T

            #1. 3D keypoints F1-Score
            TP = 1e-7 #각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이내
            FP = 1e-7 #각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이상
            FN = 1e-7 #참값이 존재하지만 키포인트 좌표가 존재하지 않는 경우(미태깅)
            for idx, gt_kpt in enumerate(gt_kpts2d):
                pred_kpt = pred_kpts2d[idx]
                if (pred_kpt == None).any():
                    FN += 1
                dist = np.linalg.norm(gt_kpt - pred_kpt)
                if dist < 50:
                    TP += 1
                elif dist >= 50:
                    FP += 1

            keypoint_precision_score = TP / (TP + FP)
            keypoint_recall_score = TP / (TP + FN)
            keypoint_f1_score = 2 * (keypoint_precision_score * keypoint_recall_score /
                                            (keypoint_precision_score + keypoint_recall_score))  # 2*TP/(2*TP+FP+FN)
            kpts_precision[camID] = keypoint_precision_score
            kpts_recall[camID] = keypoint_recall_score
            kpts_f1[camID] = keypoint_f1_score

            #2. mesh pose F1-Score
            TP = 0 #렌더링된 이미지의 각 픽셀의 segmentation 클래스(background, object, hand)가 참값(실제 RGB- segmentation map)의 클래스와 일치
            FP = 0 #렌더링된 이미지의 각 픽셀의 segmentation 클래스가 참값의 클래스와 불일치
            FN = 0 #참값이 존재하지만 키포인트 좌표의 segmentation class가 존재하지 않는 경우(미태깅) ??
            gt_seg_hand = np.squeeze((self.gt_seg[0].cpu().detach().numpy()))
            gt_seg_obj = np.squeeze((self.gt_seg_obj[0].cpu().detach().numpy()))

            TP = np.sum(np.where(hand_seg_masked > 0, hand_seg_masked == gt_seg_hand, 0)) + \
                    np.sum(np.where(obj_seg_masked > 0, obj_seg_masked == gt_seg_obj, 0))
            FP = np.sum(np.where(hand_seg_masked > 0, hand_seg_masked != gt_seg_hand, 0)) +\
                    np.sum(np.where(obj_seg_masked > 0, obj_seg_masked != gt_seg_obj, 0))
            # seg_masked_FN = (gt_seg_hand > 0) * (hand_seg_masked == 0) \
            #                 + (gt_seg_obj > 0) * (obj_seg_masked == 0)

            if TP == 0:
                TP = 1e-7
            FN = 1e-7 #np.sum(seg_masked_FN)

            mesh_seg_precision_score = TP/(TP+FP)
            mesh_seg_recall_score = TP / (TP + FN)
            mesh_seg_f1_score = 2 * (mesh_seg_precision_score * mesh_seg_recall_score /
                                        (mesh_seg_precision_score + mesh_seg_recall_score)) #2*TP/(2*TP+FP+FN)
            
            mesh_precision[camID] = mesh_seg_precision_score
            mesh_recall[camID] = mesh_seg_recall_score
            mesh_f1[camID] = mesh_seg_f1_score

            #3. hand depth accuracy
            TP = 1e-7 #각 키포인트의 렌더링된 깊이값이 참값(실제 깊이영상)의 깊이값과 20mm 이내
            FP = 1e-7 #각 키포인트의 렌더링된 깊이값이 참값(실제 깊이영상)의 깊이값과 20mm 이상
            FN = 1e-7 #참값이 존재하지만 키포인트 좌표의 깊이값이 존재하지 않는 경우(미태깅)

            #pred_kpts2d
            # obj_depth, hand_depth
            # self.gt_depth, self.gt_depth_obj
            gt_depth_hand = np.squeeze(self.gt_depth.cpu().numpy())

            gt_depth_hand[gt_depth_hand==10] = 0
            gt_depth_hand *= 100.
            hand_depth /= 10.

            # cv2.imshow("gt_depth_hand", np.array(gt_depth_hand / 100 * 255, dtype=np.uint8))
            # cv2.imshow("hand_depth", np.array(hand_depth / 100 * 255, dtype=np.uint8))
            # cv2.waitKey(0)

            for i in range(21):
                kpts2d = pred_kpts2d[i, :]

                y = np.clip(int(kpts2d[1]), 0, 479)
                x = np.clip(int(kpts2d[0]), 0, 639)
                if gt_seg_hand[y, x] == 0:
                    continue
                gt_hand_d = gt_depth_hand[y, x]
                pred_hand_d = hand_depth[y, x]
                diff = abs(gt_hand_d - pred_hand_d)
                if gt_hand_d == 0 or pred_hand_d == 0 or diff > 100:
                    FN += 1
                if diff < 20:
                    TP += 1
                else:
                    FP += 1

            if TP < 1:
                mesh_depth_precision_score = 0
                mesh_depth_recall_score = 0
                mesh_depth_f1_score = 0
            else:
                mesh_depth_precision_score = TP / (TP + FP)
                mesh_depth_recall_score = TP / (TP + FN)
                mesh_depth_f1_score = 2 * (mesh_depth_precision_score * mesh_depth_recall_score /
                                            (mesh_depth_precision_score + mesh_depth_recall_score))  # 2*TP/(2*TP+FP+FN)
            depth_precision[camID] = mesh_depth_precision_score
            depth_recall[camID] = mesh_depth_recall_score
            depth_f1[camID] = mesh_depth_f1_score

            # print("3D keypoint precision score : ", keypoint_precision_score)
            # print("3D keypoint recall score : ", keypoint_recall_score)
            # print("3D keypoint F1 score : ", keypoint_f1_score)

            # print("mesh seg precision score : ", mesh_seg_precision_score)
            # print("mesh seg recall score : ", mesh_seg_recall_score)
            # print("mesh seg F1 score : ", mesh_seg_f1_score)
            #
            # print("mesh depth precision score : ", mesh_depth_precision_score)
            # print("mesh depth recall score : ", mesh_depth_recall_score)
            # print("mesh depth F1 score : ", mesh_depth_f1_score)


        kpts_precision_avg = sum(kpts_precision.values())/len(camIdxSet)
        kpts_recall_avg = sum(kpts_recall.values())/len(camIdxSet)
        kpts_f1_avg = sum(kpts_f1.values())/len(camIdxSet)
        self.total_metrics[0] += kpts_precision_avg
        self.total_metrics[1] += kpts_recall_avg
        self.total_metrics[2] += kpts_f1_avg
        self.dfs['kpts_precision'].loc[frame] = [kpts_precision['mas'], kpts_precision['sub1'], kpts_precision['sub2'], kpts_precision['sub3'], kpts_precision_avg]
        self.dfs['kpts_recall'].loc[frame] = [kpts_recall['mas'], kpts_recall['sub1'], kpts_recall['sub2'], kpts_recall['sub3'], kpts_recall_avg]
        self.dfs['kpts_f1'].loc[frame] = [kpts_f1['mas'], kpts_f1['sub1'], kpts_f1['sub2'], kpts_f1['sub3'], kpts_f1_avg]
        mesh_precision_avg = sum(mesh_precision.values())/len(camIdxSet)
        mesh_recall_avg = sum(mesh_recall.values())/len(camIdxSet)
        mesh_f1_avg = sum(mesh_f1.values())/len(camIdxSet)
        self.total_metrics[3] += mesh_precision_avg
        self.total_metrics[4] += mesh_recall_avg
        self.total_metrics[5] += mesh_f1_avg
        self.dfs['mesh_precision'].loc[frame] = [mesh_precision['mas'], mesh_precision['sub1'], mesh_precision['sub2'], mesh_precision['sub3'], mesh_precision_avg]
        self.dfs['mesh_recall'].loc[frame] = [mesh_recall['mas'], mesh_recall['sub1'], mesh_recall['sub2'], mesh_recall['sub3'], mesh_recall_avg]
        self.dfs['mesh_f1'].loc[frame] = [mesh_f1['mas'], mesh_f1['sub1'], mesh_f1['sub2'], mesh_f1['sub3'], mesh_f1_avg]
        depth_precision_avg = sum(depth_precision.values())/len(camIdxSet)
        depth_recall_avg = sum(depth_recall.values())/len(camIdxSet)
        depth_f1_avg = sum(depth_f1.values())/len(camIdxSet)
        self.total_metrics[6] += depth_precision_avg
        self.total_metrics[7] += depth_recall_avg
        self.total_metrics[8] += depth_f1_avg
        self.dfs['depth_precision'].loc[frame] = [depth_precision['mas'], depth_precision['sub1'], depth_precision['sub2'], depth_precision['sub3'], depth_precision_avg]
        self.dfs['depth_recall'].loc[frame] = [depth_recall['mas'], depth_recall['sub1'], depth_recall['sub2'], depth_recall['sub3'], depth_recall_avg]
        self.dfs['depth_f1'].loc[frame] = [depth_f1['mas'], depth_f1['sub1'], depth_f1['sub2'], depth_f1['sub3'], depth_f1_avg]


    def save_evaluation(self, save_path, save_num):
        if save_num == 0:
            print("no data to save evaluation")
        else:
            csv_files = ['kpts_precision', 'kpts_recall', 'kpts_f1', 'mesh_precision', 'mesh_recall', 'mesh_f1', 'depth_precision', 'depth_recall', 'depth_f1']
            for idx, file in enumerate(csv_files):
                with open(os.path.join(save_path, file + '.csv'), "w", encoding='utf-8') as f:
                    ws = csv.writer(f)
                    ws.writerow(['total_avg', self.total_metrics[idx] / save_num])
                    ws.writerow(['frame', 'mas', 'sub1', 'sub2', 'sub3', 'avg'])

                df = self.dfs[file]
                df.to_csv(os.path.join(save_path, file + '.csv'), mode='a', index=True, header=False)


    def filtering_top_quality_index(self, num=60):
        metric = self.dfs['depth_f1']
        metric_len = len(metric["avg"])
        if num > metric_len:
            num = metric_len

        top_index = metric["avg"].nlargest(num).index

        return top_index

    def visualize(self, pred, pred_obj, camIdxSet, frame, save_path=None, flag_obj=False, flag_crop=False, flag_headless=False):
        flag_bb_exist = False
        for camIdx, camID in enumerate(CFG_CAMID_SET):
            if 'bb' not in self.dataloaders[camIdx][frame].keys():
                self.set_gt_nobb(camIdx, camID, frame)
                flag_bb_exist = False
            else:
                # set gt to load original input
                self.set_gt(camIdx, frame)
                flag_bb_exist = True
                # set camera status for projection

            ## HAND ##
            # project hand joint
            joints_cam = torch.unsqueeze(mano3DToCam3D(pred['joints'], self.Ms[camIdx]), 0)
            pred_kpts2d = projectPoints(joints_cam, self.Ks[camIdx])

            verts_cam = torch.unsqueeze(mano3DToCam3D(pred['verts'], self.Ms[camIdx]), 0)

            if not flag_obj:
                pred_rendered = self.cam_renderer[camIdx].render(verts_cam, pred['faces'], flag_rgb=True)
            else:
                # if flag_obj, render both objects
                verts_cam_obj = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms[camIdx]), 0)
                pred_rendered = self.cam_renderer[camIdx].render_meshes([verts_cam, verts_cam_obj],
                                                                [pred['faces'], pred_obj['faces']], flag_rgb=True)

            ## VISUALIZE ##
            rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
            depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
            seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy())
            seg_mesh = np.array(np.ceil(seg_mesh / np.max(seg_mesh)), dtype=np.uint8)

            pred_kpts2d = np.squeeze(pred_kpts2d.clone().cpu().detach().numpy())
            if flag_bb_exist:
                gt_kpts2d = np.squeeze(self.gt_kpts2d.clone().cpu().numpy())
                # check if gt kpts is nan (not detected)
                if np.isnan(gt_kpts2d).any():
                    gt_kpts2d = np.zeros((21, 2))

            rgb_2d_pred = paint_kpts(None, rgb_mesh, pred_kpts2d)
            if flag_bb_exist:
                rgb_2d_gt = paint_kpts(None, rgb_mesh, gt_kpts2d)

            if flag_crop:
                # show cropped size of input (480, 640)
                rgb_input = np.squeeze(self.gt_rgb.clone().cpu().numpy()).astype(np.uint8)
                depth_input = np.squeeze(self.gt_depth.clone().cpu().numpy())
                seg_input = np.squeeze(self.gt_seg.clone().cpu().numpy())

                # rendered image is original size (1080, 1920)
                rgb_mesh = rgb_mesh[self.bb[1]:self.bb[1]+self.bb[3], self.bb[0]:self.bb[0]+self.bb[2], :]
                depth_mesh = depth_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
                seg_mesh = seg_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]

                rgb_2d_pred = rgb_2d_pred[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2], :]
                uv1 = np.concatenate((pred_kpts2d, np.ones_like(pred_kpts2d[:, :1])), 1)
                pred_kpts2d = (self.img2bb @ uv1.T).T

                if flag_bb_exist:
                    rgb_2d_gt = rgb_2d_gt[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2], :]
                    uv1 = np.concatenate((gt_kpts2d, np.ones_like(gt_kpts2d[:, :1])), 1)
                    gt_kpts2d = (self.img2bb @ uv1.T).T
            else:
                # show original size of input (1080, 1920)
                rgb_input, depth_input, seg_input, seg_obj = self.dataloaders[CFG_CAMID_SET.index(camID)].load_raw_image(frame)

            seg_mask = np.copy(seg_mesh)
            seg_mask[seg_mesh>0] = 1
            rgb_2d_pred *= seg_mask[..., None]

            img_blend_pred = cv2.addWeighted(rgb_input, 1.0, rgb_2d_pred, 0.4, 0)

            if flag_bb_exist:
                img_blend_gt = cv2.addWeighted(rgb_input, 0.5, rgb_2d_gt, 0.7, 0)
                rgb_seg = (rgb_input * seg_input[..., None]).astype(np.uint8)
                img_blend_pred_seg = cv2.addWeighted(rgb_seg, 0.5, rgb_2d_pred, 0.7, 0)

            # create depth gap
            depth_mesh[depth_mesh==10] = 0
            depth_input[depth_input==10] = 0
            depth_gap = np.clip(np.abs(depth_input - depth_mesh)* 1000, a_min=0.0, a_max=255.0).astype(np.uint8)
            depth_gap[depth_mesh == 0] = 0

            # create seg gap, gt required
            if flag_bb_exist:
                seg_a = seg_input - seg_mesh
                seg_a[seg_a < 0] = 0
                seg_b = seg_mesh - seg_input
                seg_b[seg_b < 0] = 0
                seg_gap = ((seg_a + seg_b) * 255.0).astype(np.uint8)

            if not flag_crop:
                # resize images to (360, 640)
                img_blend_pred = cv2.resize(img_blend_pred, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
                depth_gap = cv2.resize(depth_gap, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
                if flag_bb_exist:
                    img_blend_gt = cv2.resize(img_blend_gt, dsize=(640,360), interpolation=cv2.INTER_LINEAR)
                    seg_gap = cv2.resize(seg_gap, dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
                    img_blend_pred_seg = cv2.resize(img_blend_pred_seg, dsize=(640, 360),
                                                    interpolation=cv2.INTER_LINEAR)

            blend_gt_name = "blend_gt_" + camID + "_" + str(frame)
            blend_pred_name = "blend_pred_" + camID + "_" + str(frame)
            blend_pred_seg_name = "blend_pred_seg_" + camID + "_" + str(frame)
            blend_depth_gap_name = "blend_depth_gap_" + camID + "_" + str(frame)
            blend_seg_gap_name = "blend_seg_gap_" + camID + "_" + str(frame)

            if save_path is None:
                if not flag_headless:
                    if flag_bb_exist:
                        cv2.imshow(blend_gt_name, img_blend_gt)
                    cv2.imshow(blend_pred_name, img_blend_pred)
                    # cv2.imshow(blend_pred_seg_name, img_blend_pred_seg)
                    # cv2.imshow(blend_depth_gap_name, depth_gap)
                    # cv2.imshow(blend_seg_gap_name, seg_gap)
                    cv2.waitKey(0)
                else:
                    if flag_bb_exist:
                        cv2.imwrite(os.path.join("./for_headless_server", blend_gt_name + '.png'), img_blend_gt)
                    cv2.imwrite(os.path.join("./for_headless_server", blend_pred_name + '.png'), img_blend_pred)
                    cv2.imwrite(os.path.join("./for_headless_server", blend_depth_gap_name + '.png'), depth_gap)
            else:
                save_path_cam = os.path.join(save_path, camID)
                os.makedirs(save_path_cam, exist_ok=True)
                cv2.imwrite(os.path.join(save_path_cam, blend_pred_name + '.png'), img_blend_pred)
                # cv2.imwrite(os.path.join(save_path_cam, blend_pred_seg_name + '.png'), img_blend_pred_seg)
                # cv2.imwrite(os.path.join(save_path_cam, blend_depth_gap_name + '.png'), depth_gap)

                # save meshes
                if CFG_SAVE_MESH:
                    hand_verts = mano3DToCam3D(pred['verts'], self.Ms[camIdx])
                    hand = trimesh.Trimesh(hand_verts.detach().cpu().numpy(), pred['faces'][0].detach().cpu().numpy())
                    hand.export(os.path.join(save_path_cam, f'mesh_hand_{camID}_{frame}.obj'))

                    if flag_obj:
                        obj_verts = mano3DToCam3D(pred_obj['verts'], self.Ms[camIdx])
                        obj = trimesh.Trimesh(obj_verts.detach().cpu().numpy(), pred_obj['faces'][0].detach().cpu().numpy())
                        obj.export(os.path.join(save_path_cam, f'mesh_obj_{camID}_{frame}.obj'))

                # create contact map
                hand_pcd = Pointclouds(points=verts_cam)
                obj_mesh = Meshes(verts=verts_cam_obj.detach(),
                                  faces=pred_obj['faces'])  # optimize only hand meshes
                inter_dist = point_mesh_face_distance(obj_mesh, hand_pcd)
                # debug = inter_dist.clone().cpu().detach().numpy()
                contact_mask = inter_dist < CFG_CONTACT_DIST_VIS
                contact_map = torch.ones(778).cuda() * -1.
                contact_map[contact_mask] = inter_dist[contact_mask]

                pred['contact'] = contact_map.clone().detach()

                # vis contact map
                if CFG_VIS_CONTACT:
                    contact_idx = torch.where(contact_map > 0)
                    if not contact_idx[0].nelement() == 0:
                        max = contact_map[contact_idx].max()
                        contact_map[contact_idx] = contact_map[contact_idx] / max
                        contact_map[contact_idx] = 1 - contact_map[contact_idx]
                    contact_map[contact_map == -1.] = 0.

                    save_path_contactmap = os.path.join(save_path, 'contactmap')
                    os.makedirs(save_path_contactmap, exist_ok=True)
                    pcd = o3d.geometry.PointCloud()
                    hand_verts = mano3DToCam3D(pred['verts'], self.Ms[camIdx])
                    pcd.points = o3d.utility.Vector3dVector(hand_verts.detach().cpu().numpy())
                    colors = contact_map.unsqueeze(1).detach().cpu().numpy()
                    colors = np.concatenate([colors, colors, 0.5-colors*0.5], axis=1)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_point_cloud(os.path.join(save_path_contactmap, f"contact_{frame}_{camID}.ply"), pcd)

                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
                    vis.get_render_option().point_size = 15.0
                    vis.add_geometry(pcd)
                    vis.capture_screen_image(os.path.join(save_path_contactmap, f"contact_{frame}_{camID}.jpg"), do_render=True)
                    vis.destroy_window()
