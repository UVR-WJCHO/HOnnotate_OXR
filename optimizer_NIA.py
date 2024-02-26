import os
import sys
import warnings
warnings.filterwarnings(action='ignore')

from utils.lossUtils import *

import time
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from absl import logging


camIDset = ['mas', 'sub1', 'sub2', 'sub3']


def mano3DToCam3D(xyz3D, ext):
    device = xyz3D.device
    xyz3D = torch.squeeze(xyz3D)
    ones = torch.ones((xyz3D.shape[0], 1), device=device)

    xyz4Dcam = torch.cat([xyz3D, ones], axis=1)
    # world to target cam
    xyz3Dcam2 = xyz4Dcam @ ext.T  # [:, :3]

    return xyz3Dcam2


class MultiViewLossFunc_OBJ(nn.Module):
    def __init__(self, device, bs=1, dataloaders=None, renderers=None, losses=None):
        super(MultiViewLossFunc_OBJ, self).__init__()
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
        self.default_zero = torch.tensor([0.0], requires_grad=True).to(self.device)

        self.Ks = dataloaders.Ks_list
        self.Ms = dataloaders.Ms_list

        self.const = Constraints()
        self.rot_min = torch.tensor(np.asarray(params.rot_min_list)).to(self.device)
        self.rot_max = torch.tensor(np.asarray(params.rot_max_list)).to(self.device)

    def set_object_main_extrinsic(self, obj_main_cam_idx):
        self.main_Ms_obj = self.Ms[obj_main_cam_idx]

    def get_pose_constraint_tensor(self):
        pose_mean_tensor = torch.tensor(params.pose_mean_list).cuda()
        pose_reg_tensor = torch.tensor(params.pose_reg_list).cuda()
        return pose_reg_tensor, pose_mean_tensor

    def compute_reg_loss(self, mano_tensor, pose_mean_tensor, pose_reg_tensor):
        reg_loss = ((mano_tensor - pose_mean_tensor) ** 2) * pose_reg_tensor
        return torch.sum(reg_loss, -1)


    def set_gt(self, frame, camIdx):
        gt_sample = self.dataloaders[frame, camIdx]
        self.gt_rgb = gt_sample['rgb']
        # self.gt_depth = gt_sample['depth']
        self.gt_depth_obj = gt_sample['depth_obj']

        # self.gt_seg = gt_sample['seg']
        self.gt_seg_obj = gt_sample['seg_obj']
        self.bb = gt_sample['bb']


    def set_gt_raw_depth(self, camIdx, frame):
        _, self.gt_depth_raw, _, _ = self.dataloaders[camIdx].load_raw_image(frame)

    def set_main_cam(self, main_cam_idx=0):
        # main_cam_params = self.dataloaders[main_cam_idx].cam_parameter
        self.main_Ks = self.Ks[main_cam_idx]
        self.main_Ms = self.Ms[main_cam_idx]

    def forward(self, pred_obj, camIdxSet, frame, loss_dict):

        self.loss_dict = loss_dict

        verts_obj_set = {}
        pred_obj_render_set = {}

        ## compute per cam predictions
        for camIdx in camIdxSet:
            verts_obj_cam = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms[camIdx]), 0)
            pred_obj_rendered = self.cam_renderer[camIdx].render(verts_obj_cam, pred_obj['faces'])

            verts_obj_set[camIdx] = verts_obj_cam
            pred_obj_render_set[camIdx] = pred_obj_rendered

        ## compute losses per cam
        losses_cam = {}
        for camIdx in camIdxSet:
            loss = {}

            self.set_gt(frame, camIdx)

            pred_obj_rendered = pred_obj_render_set[camIdx]
            if 'seg_obj' in self.loss_dict:
                pred_seg_obj = pred_obj_rendered['seg'][:, self.bb[1]:self.bb[1] + self.bb[3],
                               self.bb[0]:self.bb[0] + self.bb[2]]
                pred_seg_obj = torch.div(pred_seg_obj, torch.max(pred_seg_obj))
                pred_seg_obj = torch.ceil(pred_seg_obj)
                # pred_seg_obj[pred_seg_obj == 0] = 10.

                # a = np.squeeze(self.gt_seg_obj.clone().cpu().detach().numpy())
                # b = np.squeeze(pred_seg_obj.clone().cpu().detach().numpy())
                # cv2.imshow("gt seg obj", a)
                # cv2.imshow("pred seg obj", b)
                # cv2.waitKey(0)

                self.cam_renderer[camIdx].register_seg(self.gt_seg_obj)
                loss_seg_obj, seg_obj_gap = self.cam_renderer[camIdx].compute_seg_loss(pred_seg_obj)

                loss['seg_obj'] = loss_seg_obj * 1e0

                seg_obj_gap = np.squeeze((seg_obj_gap[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                cv2.imshow("seg_obj_gap"+str(camIdx), seg_obj_gap)
                cv2.waitKey(1)

                # if camIdx == 0:
                #     pred_seg_obj = np.squeeze((pred_seg_obj[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                #     seg_obj_gap = np.squeeze((seg_obj_gap[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                #     gt_seg_obj = np.squeeze((self.gt_seg_obj[0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                #
                #     cv2.imshow("pred_seg_obj", pred_seg_obj)
                #     cv2.imshow("gt_seg_obj", gt_seg_obj)
                #     cv2.imshow("seg_obj_gap", seg_obj_gap)
                #     cv2.waitKey(1)

            if 'depth_obj' in self.loss_dict:
                pred_depth_obj = pred_obj_rendered['depth'][:, self.bb[1]:self.bb[1] + self.bb[3],
                                 self.bb[0]:self.bb[0] + self.bb[2]]

                # a = np.squeeze(self.gt_depth_obj.clone().cpu().detach().numpy())
                # b = np.squeeze(pred_depth_obj.clone().cpu().detach().numpy())
                # cv2.imshow("gt depth obj", a)
                # cv2.imshow("pred depth obj", b)
                # cv2.waitKey(0)

                depth_ref = self.gt_depth_obj.clone()
                # depth_ref[depth_ref == 10.0] = 0
                # depth_ref *= 100.0

                self.cam_renderer[camIdx].register_depth(depth_ref)
                loss_depth_obj, depth_obj_gap = self.cam_renderer[camIdx].compute_depth_loss(pred_depth_obj)

                # depth_obj_gap = torch.abs(pred_depth_obj - self.gt_depth_obj)
                # depth_obj_gap[self.gt_depth_obj == 0] = 0
                # loss_depth_obj = torch.mean(depth_obj_gap.view(self.bs, -1), -1)

                loss['depth_obj'] = loss_depth_obj * 1e-1

                # pred_depth = np.squeeze(pred_depth_obj[0].cpu().detach().numpy())
                # pred_depth[pred_depth == 10.] = 0
                # pred_depth_vis = (pred_depth * 100).astype(np.uint8)
                # cv2.imshow("pred_depth"+str(camIdx), pred_depth_vis)

                # gt_depth = np.squeeze(self.gt_depth_obj[0].cpu().detach().numpy())
                # gt_depth[gt_depth == 10.] = 0
                # gt_depth_vis = (gt_depth * 100).astype(np.uint8)
                # cv2.imshow("gt_depth_vis"+str(camIdx), gt_depth_vis)

                depth_gap_vis = np.squeeze((depth_obj_gap[0].cpu().detach().numpy())*10).astype(np.uint8)
                cv2.imshow("depth_gap_vis"+str(camIdx), depth_gap_vis)
                cv2.waitKey(1)

                # if camIdx == 0:
                #     pred_depth = np.squeeze(pred_depth_obj[0].cpu().detach().numpy())
                #     pred_depth[pred_depth==10.] = 0
                #     gt_depth = np.squeeze(self.gt_depth_obj[0].cpu().detach().numpy())
                #     gt_depth[gt_depth == 10.] = 0
                #     pred_depth_vis = (pred_depth*100).astype(np.uint8)
                #     gt_depth_vis = (gt_depth * 100).astype(np.uint8)
                #     depth_gap_vis = np.squeeze((depth_obj_gap[0].cpu().detach().numpy())).astype(np.uint8)
                #     cv2.imshow("pred_depth", pred_depth_vis)
                #     cv2.imshow("gt_depth_vis", gt_depth_vis)
                #     cv2.imshow("depth_gap_vis", depth_gap_vis)
                #     cv2.waitKey(1)

            losses_cam[camIdx] = loss

        return losses_cam


    def visualize(self, pred_obj, cams, frame):

        for camIdx in cams:
            self.set_gt(frame, camIdx)
            camID = camIDset[camIdx]

            verts_cam_obj = torch.unsqueeze(mano3DToCam3D(pred_obj['verts'], self.Ms[camIdx]), 0)
            pred_rendered = self.cam_renderer[camIdx].render(verts_cam_obj, pred_obj['faces'], flag_rgb=True)

            ## VISUALIZE ##
            rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
            depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
            seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy())
            seg_mesh = np.array(np.ceil(seg_mesh / np.max(seg_mesh)), dtype=np.uint8)

            # show cropped size of input (480, 640)
            rgb_input = np.squeeze(self.gt_rgb.clone().cpu().numpy()).astype(np.uint8)
            # depth_input = np.squeeze(self.gt_depth.clone().cpu().numpy())
            # seg_input = np.squeeze(self.gt_seg.clone().cpu().numpy())

            # rendered image is original size (1080, 1920)
            rgb_mesh = rgb_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2], :]
            depth_mesh = depth_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]
            seg_mesh = seg_mesh[self.bb[1]:self.bb[1] + self.bb[3], self.bb[0]:self.bb[0] + self.bb[2]]

            seg_mask = np.copy(seg_mesh)
            seg_mask[seg_mesh > 0] = 1

            ### create depth gap
            # depth_mesh[depth_mesh == 10] = 0
            # depth_input[depth_input == 10] = 0
            # depth_gap = np.clip(np.abs(depth_input - depth_mesh) * 1000, a_min=0.0, a_max=255.0).astype(np.uint8)
            # depth_gap[depth_mesh == 0] = 0

            img_blend_pred = cv2.addWeighted(rgb_input, 0.3, rgb_mesh, 0.8, 0)

            blend_gt_name = "blend_gt_" + camID
            blend_pred_name = "blend_pred_" + camID
            blend_pred_seg_name = "blend_pred_seg_" + camID
            blend_depth_gap_name = "blend_depth_gap_" + camID
            blend_seg_gap_name = "blend_seg_gap_" + camID

            cv2.imshow(blend_pred_name, img_blend_pred)
            # cv2.imshow(blend_pred_seg_name, img_blend_pred_seg)
            # cv2.imshow(blend_depth_gap_name, depth_gap)
            # cv2.imshow(blend_seg_gap_name, seg_gap)
            cv2.waitKey(1)


class ObjModel(nn.Module):
    def __init__(self, device, batch_size, obj_template, obj_init_rot=np.eye(3)):
        super(ObjModel, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.template_verts, self.template_faces = obj_template['verts'], obj_template['faces']

        obj_rot = torch.FloatTensor(np.eye(3)).unsqueeze(0)
        obj_rot = matrix_to_axis_angle(obj_rot) #[1, 3]
        obj_trans = torch.zeros((1, 3))

        self.obj_rot = nn.Parameter(obj_rot.to(self.device))
        self.obj_trans = nn.Parameter(obj_trans.to(self.device))

        self.h = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).to(self.device)


    def sixd2matrot(self, pose_6d):
        '''
        :param pose_6d: Nx6
        :return: pose_matrot: Nx3x3
        '''
        rot_vec_1 = pose_6d[:, :3]
        rot_vec_2 = pose_6d[:, 3:6]
        rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2)
        pose_matrot = torch.stack([rot_vec_1, rot_vec_2, rot_vec_3], dim=-1)
        return pose_matrot

    def matrot2sixd(self, pose_matrot):
        '''
        :param pose_matrot: Nx3x3
        :return: pose_6d: Nx6
        '''
        pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
        return pose_6d

    def get_object_mat(self):
        obj_pose = torch.cat((self.obj_rot, self.obj_trans), -1)

        obj_rot = axis_angle_to_matrix(obj_pose[:, :3]).squeeze()
        obj_trans = obj_pose[:, 3:].T
        obj_pose = torch.cat((obj_rot, obj_trans), -1)

        obj_mat = torch.cat([obj_pose, self.h], dim=0)

        return np.squeeze(obj_mat.detach().cpu().numpy())

    def update_pose(self, pose):
        obj_rot = torch.unsqueeze(torch.tensor(pose[:, :3], dtype=torch.float32), 0)
        # obj_rot = self.matrot2sixd(obj_rot)
        obj_rot = matrix_to_axis_angle(obj_rot)
        obj_rot = obj_rot.view(self.batch_size, -1)

        obj_trans = torch.tensor(pose[:, 3:], dtype=torch.float32).T
        # obj_pose = torch.cat((obj_rot, obj_trans), -1)

        self.obj_rot = nn.Parameter(obj_rot.to(self.device))
        self.obj_trans = nn.Parameter(obj_trans.to(self.device))
        # self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_rot.requires_grad = True
        self.obj_trans.requires_grad = True

    def change_grads(self, rot=False, trans=False):
        self.obj_rot.requires_grad = rot
        self.obj_trans.requires_grad = trans

    def apply_transform(self, obj_rot, obj_trans, obj_verts):
        obj_rot = axis_angle_to_matrix(obj_rot).squeeze()
        obj_trans = obj_trans.T
        obj_pose = torch.cat((obj_rot, obj_trans), -1)
        obj_mat = torch.cat([obj_pose, self.h], dim=0)

        # Append 1 to each coordinate to convert them to homogeneous coordinates
        h = torch.ones((obj_verts.shape[0], 1), device=self.device)
        homogeneous_points = torch.cat((obj_verts, h), 1)

        # Apply matrix multiplication
        transformed_points = homogeneous_points @ obj_mat.T
        # Convert back to Cartesian coordinates
        transformed_points_cartesian = transformed_points[:, :3] / transformed_points[:, 3:]
        transformed_points_cartesian = transformed_points_cartesian.view(self.batch_size, -1, 3)

        return transformed_points_cartesian

    def forward(self):
        obj_pose = [self.obj_rot, self.obj_trans]
        obj_verts = self.apply_transform(self.obj_rot, self.obj_trans, self.template_verts) #/ 10.0

        return {'verts': obj_verts, 'faces': self.template_faces, 'pose': obj_pose}


def optimize_obj(model_obj, loss_func, detected_cams, frame, lr_init_obj, seq, trialName, target_iter=100, flag_vis=False):
    kps_loss = {}
    use_contact_loss = False
    use_penetration_loss = False

    # set initial loss, early stopping threshold
    best_loss = torch.inf
    prev_kps_loss = 0
    prev_obj_loss = torch.inf
    prev_depthseg_loss = 0
    early_stopping_patience = 0
    early_stopping_patience_v2 = 0
    early_stopping_patience_obj = 0

    loss_weight = CFG_LOSS_WEIGHT

    print("start trans update")
    optimizer_obj = initialize_optimizer_obj(model_obj, lr_rot=0, lr_trans=lr_init_obj)
    lr_scheduler_obj = torch.optim.lr_scheduler.StepLR(optimizer_obj, step_size=5, gamma=0.95)
    model_obj.change_grads(rot=False, trans=True)

    for iter in range(30):
        t_iter = time.time()

        optimizer_obj.zero_grad()
        loss_all = {'kpts2d': 0.0, 'depth': 0.0, 'seg': 0.0, 'reg': 0.0, 'contact': 0.0, 'penetration': 0.0,
                    'depth_rel': 0.0, 'temporal': 0.0, 'kpts_tip': 0.0, 'depth_obj': 0.0, 'seg_obj': 0.0,
                    'pose_obj': 0.0}

        obj_param = model_obj()

        losses = loss_func(pred_obj=obj_param, camIdxSet=detected_cams, frame=frame, loss_dict=CFG_LOSS_DICT)

        if flag_vis:
            loss_func.visualize(pred_obj=obj_param, cams=detected_cams, frame=frame)

        ## apply cam weight
        for camIdx in detected_cams:
            loss_cam = losses[camIdx]
            for key in loss_cam.keys():
                loss_all[key] += loss_cam[key] #* float(CFG_CAM_WEIGHT[camIdx])

        ## apply loss weight
        total_loss = sum(loss_all[k] * loss_weight[k] for k in CFG_LOSS_DICT) / len(detected_cams)
        total_loss.backward(retain_graph=False)

        optimizer_obj.step()
        lr_scheduler_obj.step()

        # iter_t = time.time() - t_iter
        # print("iter_t : ", iter_t)
        logs = ["[{} - {} - frame {}] [All] Iter: {}, Loss: {:.4f}".format(seq, trialName, frame, iter,
                                                                           total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                 key in CFG_LOSS_DICT]
        # logging.info(''.join(logs))
        print(logs)


    print("start all update")
    optimizer_obj = initialize_optimizer_obj(model_obj, lr_rot=lr_init_obj * 5e-3, lr_trans=lr_init_obj*0.8)
    lr_scheduler_obj = torch.optim.lr_scheduler.StepLR(optimizer_obj, step_size=5, gamma=0.95)
    model_obj.change_grads(rot=True, trans=True)
    for iter in range(target_iter):
        t_iter = time.time()

        optimizer_obj.zero_grad()
        loss_all = {'kpts2d': 0.0, 'depth': 0.0, 'seg': 0.0, 'reg': 0.0, 'contact': 0.0, 'penetration': 0.0,
                    'depth_rel': 0.0, 'temporal': 0.0, 'kpts_tip': 0.0, 'depth_obj': 0.0, 'seg_obj': 0.0,
                    'pose_obj': 0.0}

        obj_param = model_obj()

        losses = loss_func(pred_obj=obj_param, camIdxSet=detected_cams, frame=frame, loss_dict=CFG_LOSS_DICT)

        if flag_vis:
            loss_func.visualize(pred_obj=obj_param, cams=detected_cams, frame=frame)

        ## apply cam weight
        for camIdx in detected_cams:
            loss_cam = losses[camIdx]
            for key in loss_cam.keys():
                loss_all[key] += loss_cam[key] #* float(CFG_CAM_WEIGHT[camIdx])

        ## apply loss weight
        total_loss = sum(loss_all[k] * loss_weight[k] for k in CFG_LOSS_DICT) / len(detected_cams)
        total_loss.backward(retain_graph=True)

        optimizer_obj.step()
        lr_scheduler_obj.step()

        # iter_t = time.time() - t_iter
        # print("iter_t : ", iter_t)
        logs = ["[{} - {} - frame {}] [All] Iter: {}, Loss: {:.4f}".format(seq, trialName, frame, iter, total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if key in CFG_LOSS_DICT]
        # logging.info(''.join(logs))
        print(logs)


