import sys
import numpy as np
import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import modules.config as cfg
import modules.NIA_utils as NIA_utils
import cv2



def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    uv = torch.matmul(K, xyz.transpose(2, 1)).transpose(2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]

def compute_reg_loss(mano_tensor, pose_mean_tensor, pose_reg_tensor):
    reg_loss = ((mano_tensor - pose_mean_tensor)**2)*pose_reg_tensor
    return reg_loss



class Model(nn.Module):
    def __init__(self, mano_path, device, batch_size, root_idx=0):
        super().__init__()

        self.device = device
        self.change_render_setting(False)
        self.renderer_d = None
        self.renderer_col = None
        self.wrist_idx = 0
        self.mcp_idx = 9
        self.root_idx = root_idx

        self.key_bone_len = 10.0


        self.mano_layer = ManoLayer(center_idx = self.root_idx, flat_hand_mean=False,
            side="right", mano_root=mano_path, ncomps=45, use_pca=False, root_rot_mode="axisang",
            joint_rot_mode="axisang").cuda()
        self.img_center_width = cfg.IMG_WIDTH / 2.0    # 940
        self.img_center_height = cfg.IMG_HEIGHT / 2.0  # 540
        self.batch_size = batch_size

        self.K = None
        self.R = None
        self.T = None
        self.K_main = None
        self.R_main = None
        self.T_main = None
        self.extrinsic = None
        self.extrinsic_main = None

        self.kpts_2d_idx_set = set()
        self.kpts_2d_ref = torch.zeros((self.batch_size, 21, 2)).cuda()
        self.kpts_3d_leap = torch.zeros((21, 3)).cuda()

        ################# Initial pose for banana seq 0000.png
        self.xy_root = nn.Parameter(torch.tensor([9.8692, -2.4945], dtype=torch.float32).cuda().repeat(self.batch_size, 1))
        self.z_root = nn.Parameter(torch.tensor([55], dtype=torch.float32).cuda().repeat(self.batch_size, 1))


        self.is_rot_only = False
        self.input_rot = nn.Parameter(NIA_utils.clip_mano_hand_rot(torch.zeros(self.batch_size, 3).cuda()))
        self.input_pose = nn.Parameter(torch.zeros(self.batch_size, 45).cuda())
        self.input_shape = nn.Parameter(torch.zeros(self.batch_size, 10).cuda())
        self.shape_adjusted = NIA_utils.clip_mano_hand_shape(self.input_shape)

        self.camera_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda().repeat(self.batch_size, 1)
        self.ups = torch.tensor([0.0,-1.0,-1.0], dtype=torch.float32).cuda().repeat(self.batch_size, 1)
        self.ats = torch.tensor([0.0,0.0,-1.0], dtype=torch.float32).cuda().repeat(self.batch_size, 1)

        self.image_render_rgb = None
        self.image_render_d = None

        self.change_grads(root=False, rot=False, pose=False, shape=False)
        self.is_root_only = False

        self.is_loss_leap3d = False
        self.set_up_camera()    # camera for projecting mano model joints, different with camera for rendering mesh

        ################# Initial pose for banana seq 0000.png
        self.input_rot += torch.tensor([[2.8380e+00, -3.1890e+00,  5.0351e-01]]).cuda()
        self.input_pose += torch.tensor([[3.3100e-02, -3.0461e-01,
         -9.8433e-01,  0.0000e+00,  0.0000e+00, -9.2932e-01,  0.0000e+00,
          0.0000e+00,  1.0254e+00,  1.2304e-03, -1.8089e-01, -8.8209e-01,
          0.0000e+00,  0.0000e+00, -4.7442e-01,  0.0000e+00,  0.0000e+00,
          7.1177e-01, -1.1519e-02,  3.7421e-02, -1.2500e+00,  9.0822e-02,
          0.0000e+00, -6.4365e-01,  0.0000e+00,  0.0000e+00,  1.5000e+00,
         -2.6237e-03, -3.4252e-01, -1.0000e+00,  0.0000e+00,  0.0000e+00,
         -2.7805e-01,  0.0000e+00,  0.0000e+00,  7.6250e-01, -5.0000e-01,
         -2.6040e-01,  1.1759e-01,  0.0000e+00, -9.1029e-02,  0.0000e+00,
          0.0000e+00, -5.8370e-02, -5.0000e-01]]).cuda()

        # Inital set up
        self.pose_reg_tensor, self.pose_mean_tensor = NIA_utils.get_pose_constraint_tensor()
        self.pose_adjusted = self.input_pose
        self.pose_adjusted_all = torch.cat((self.input_rot, self.pose_adjusted), 1)


        # normalize scale
        hand_verts, hand_joints = self.mano_layer(self.pose_adjusted_all, self.shape_adjusted)
        self.scale = torch.tensor([[self.compute_normalized_scale(hand_joints)]]).cuda()

        self.init_loss_mode()

        self.h = torch.tensor([[0, 0, 0, 1]]).cuda()
    """
    def set_kpts_3d_glob_leap(self, data):
        self.kpts_3d_glob_leap = torch.from_numpy(data).cuda()[1:, :].reshape((21, 3)) # ignore palm joint
        self.set_xyz_root(self.kpts_3d_glob_leap[self.root_idx:self.root_idx+1, [1, 0, 2]])
        
    def set_kpts_3d_glob_leap_no_palm(self, data, with_xyz_root = False):
        self.kpts_3d_glob_leap = torch.from_numpy(data).cuda().reshape((21, 3))
        if not with_xyz_root:
            self.set_xyz_root(self.kpts_3d_glob_leap[self.root_idx:self.root_idx+1, [1, 0, 2]])
        else:
            self.kpts_3d_glob_leap = self.kpts_3d_glob_leap + torch.cat((self.xy_root, self.z_root), -1)
    
    """
    def set_segmentmap(self, image_seg_gt):
        self.image_seg, _ = torch.max(torch.tensor(image_seg_gt[None,...] > 128, dtype=torch.float32).cuda(), -1)

    def set_depthmap(self, image_depth_gt):
        self.image_depth = torch.tensor(image_depth_gt, dtype=torch.float32).cuda()

    def set_bbox(self, bbox):
        self.bbox = bbox


    def set_renderer(self, renderer_d, renderer_col):
        self.renderer_d = renderer_d
        self.renderer_col = renderer_col

    def set_kpts_2d_gt_val(self, kpt_2d_idx, kpt_2d_val):
        self.kpts_2d_idx_set.add(kpt_2d_idx)
        self.kpts_2d_ref[:, kpt_2d_idx] = torch.tensor(kpt_2d_val).cuda()

    def set_kpts_2d_gt(self, kpt_2d_gt):
        self.kpts_2d_ref = torch.unsqueeze(torch.tensor(kpt_2d_gt), 0).cuda()

    def set_pose_prev(self, pose_prev):
        self.pose_prev = pose_prev.clone().detach()

    def set_xyz_root(self, xyz_root):
        xyz_root = xyz_root.clone().detach().view(self.batch_size, -1)
        self.xy_root.data = xyz_root[:,:2]
        self.z_root.data = xyz_root[:,2:3]

    def set_input_rot(self, input_rot):
        input_rot = input_rot.view(self.batch_size, -1)
        self.input_rot.data = input_rot.clone().detach()
        
    def set_rot_only(self):
        self.is_rot_only = True
        
    def set_root_only(self):
        self.is_root_only = True

    def set_input_shape(self, input_shape):
        input_shape = input_shape.view(self.batch_size, -1)
        self.input_shape.data = input_shape.clone().detach()
        self.shape_adjusted = self.input_shape

    def set_input_pose(self, input_pose):
        input_pose = input_pose.view(self.batch_size, -1)
        self.input_pose.data = input_pose.clone().detach()


    def reset_kpts_2d(self):
        self.kpts_2d_idx_set = set()
        self.kpts_2d_ref[...] = 0.0

    def reset_parameters(self, keep_mano = False):
        if not keep_mano:
            self.xy_root.data = torch.tensor([0.0, 0.0], dtype=torch.float32).cuda().repeat(self.batch_size, 1)
            self.z_root.data = torch.tensor([50], dtype=torch.float32).cuda().repeat(self.batch_size, 1)
            self.input_rot.data = utils.clip_mano_hand_rot(torch.zeros(self.batch_size, 3).cuda())
            self.input_pose.data = torch.zeros(self.batch_size, 45).cuda()
            self.input_shape.data = torch.zeros(self.batch_size, 10).cuda()
            self.shape_adjusted = self.input_shape
        self.kpts_2d_idx_set = set()
        self.kpts_2d_ref[...] = 0.0
        self.kpts_3d_leap[...] = 0.0
        self.rot_freeze_state = False
        self.rot_freeze_value = None
        self.finger_freeze_state_list = [False for i in range(5)]
        self.finger_freeze_pose_list = [None for i in range(5)]
        self.reset_param_grads()

    def reset_param_grads(self):
        self.change_grads(root=False, rot=False, pose=False, shape=False)
        self.is_rot_only = False
        self.is_root_only = False

    def set_cam_params(self, intrinsic, extrinsic, flag_main=False):

        K = torch.tensor(intrinsic, dtype=torch.float32).cuda()

        Rot = torch.FloatTensor(extrinsic[:, :-1]).cuda()
        Trans = torch.unsqueeze(torch.FloatTensor(extrinsic[:, -1]), -1).cuda()



        ext = torch.cat([Rot, Trans], axis=-1)
        Rot = torch.unsqueeze(Rot, 0)
        Trans = Trans.T

        if flag_main:
            self.K_main = K
            self.R_main = Rot
            self.T_main = Trans
            self.extrinsic_main = ext
        else:
            self.K = K
            self.R = Rot
            self.T = Trans
            self.extrinsic = ext

    def set_up_camera(self):
        self.R = look_at_rotation(self.camera_position, at=self.ats, up=self.ups, device=self.device)
        self.T = -torch.bmm(self.R.transpose(1, 2), self.camera_position[:,:,None])[:, :, 0]


    def change_render_setting(self, new_state):
        self.is_rendering = new_state

    def change_grads(self, root=False, rot=False, pose=False, shape=False):
        self.xy_root.requires_grad = root
        self.z_root.requires_grad = root
        self.input_rot.requires_grad = rot
        self.input_pose.requires_grad = pose
        self.input_shape.requires_grad = shape


    def init_loss_mode(self):
        self.set_loss_mode(False, False, False)

    def set_loss_mode(self, is_loss_seg, is_loss_2d, is_loss_reg, is_loss_depth=False):
        self.is_loss_seg = is_loss_seg
        self.is_loss_2d = is_loss_2d
        self.is_loss_reg = is_loss_reg
        self.is_loss_depth = is_loss_depth

    def get_mano_numpy(self):
        # concatenate the xyz_root, input_rot, input_pose so it's 45 + 3 + 3 = 51
        a_1 = self.xy_root.data.detach().clone().cpu().numpy().reshape((-1))
        a_2 = self.z_root.data.detach().clone().cpu().numpy().reshape((-1))
        b = self.pose_adjusted_all.detach().clone().cpu().numpy().reshape((-1))
        return np.concatenate((a_1, a_2, b))
        
    def compute_normalized_scale(self, hand_joints):
        return (torch.sum((hand_joints[0, self.mcp_idx] - hand_joints[0, self.wrist_idx])**2)**0.5)/self.key_bone_len

    def mano3DToCam3D(self, xyz3D):
        xyz3D = torch.squeeze(xyz3D)

        ones = torch.ones((xyz3D.shape[0], 1)).cuda()

        # mano to world
        xyz4Dcam = torch.concatenate([xyz3D, ones], axis=1)
        projMat = torch.concatenate((self.extrinsic_main, self.h), 0)
        xyz4Dworld = (torch.linalg.inv(projMat) @ xyz4Dcam.T).T

        # world to target cam
        xyz3Dcam2 = xyz4Dworld @ self.extrinsic.T  # [:, :3]

        return xyz3Dcam2


    def forward_basic(self):
        # Obtain verts and joints locations using MANOLayer
        self.pose_adjusted.data = NIA_utils.clip_mano_hand_pose(self.input_pose)
        self.pose_adjusted_all = torch.cat((self.input_rot, self.pose_adjusted), 1)
        hand_verts, hand_joints = self.mano_layer(self.pose_adjusted_all, self.shape_adjusted)

        # Create xzy_root
        self.xyz_root = torch.cat((self.xy_root, self.z_root), -1)

        # Shifting & scaling
        hand_joints = hand_joints/(self.scale)
        hand_joints += self.xyz_root[:,None,:]
        self.verts = hand_verts/(self.scale)
        self.verts += self.xyz_root[:,None,:]

        coordChangeMat_py3D = torch.tensor([[-1., 0., 0.], [0, -1., 0.], [0., 0., 1.]], dtype=torch.float32).cuda()
        hand_joints = hand_joints @ coordChangeMat_py3D.T
        self.verts = self.verts @ coordChangeMat_py3D.T

        # including extrinsic transformation
        self.verts = torch.unsqueeze(self.mano3DToCam3D(self.verts), axis=0)
        hand_joints = torch.unsqueeze(self.mano3DToCam3D(hand_joints), axis=0)      # to (1080, 1920) size uv (same as renderer)

        # self.kpts_3d = hand_joints
        uv_mano_full = projectPoints(hand_joints, self.K)

        uv_mano_full[torch.isnan(uv_mano_full)] = 0.0
        self.kpts_2d = uv_mano_full.clone().detach()
        
        return hand_joints, uv_mano_full

    def forward(self):      # self.kpts_3d : mano 3D kpts , kpts_3d_leap : predicted 3D kpts
        hand_joints, uv_mano_full = self.forward_basic()


        if self.is_rendering or self.is_loss_seg or self.is_loss_depth:
            faces = self.mano_layer.th_faces.repeat(self.batch_size, 1, 1)
            verts_rgb = torch.ones_like(self.verts)
            textures = TexturesVertex(verts_features=verts_rgb)
            hand_mesh = Meshes(
                verts=self.verts,
                faces=faces,
                textures=textures
            )
            # self.image_render_rgb = self.renderer_col(meshes_world=hand_mesh, R=self.R, T=self.T)
            # self.image_render_d = self.renderer_d(meshes_world=hand_mesh, R=self.R, T=self.T).zbuf

            self.image_render_rgb = self.renderer_col(meshes_world=hand_mesh)
            self.image_render_d = self.renderer_d(meshes_world=hand_mesh).zbuf

            # don't know why, rendered image is fliped x, y axis. 2dKpts & verts has same coordinate. only projection/rendered coord is different.
            self.image_render_rgb = torch.flip(self.image_render_rgb, [1, 2])
            self.image_render_d = torch.flip(self.image_render_d, [1, 2])

            # Calculate the silhouette loss
        if self.is_loss_seg:
            loss_seg = torch.sum(((self.image_render_d[..., 3] - self.image_seg) ** 2).view(self.batch_size, -1), -1)
        else:
            loss_seg = torch.tensor(0.0)

        # Calculate the 2D loss
        if self.is_loss_2d:
            if not self.is_root_only:
                # MSE
                loss_2d = torch.sum(((uv_mano_full - self.kpts_2d_ref) ** 2).reshape(self.batch_size, -1), -1)
            else:
                # MSE
                loss_2d = torch.sum(((uv_mano_full[:, [0], :] - self.kpts_2d_ref[:, [0], :]) ** 2).\
                    view(self.batch_size, -1), -1)
        else:
            loss_2d = torch.tensor(0.0)

        # compute regularization loss       -- self.pose_adjusted value 변하는지 check.
        if self.is_loss_reg:
            loss_reg = compute_reg_loss(self.pose_adjusted, self.pose_mean_tensor, self.pose_reg_tensor)
        else:
            loss_reg = torch.tensor(0.0)

        if self.is_loss_depth:
            bb = self.bbox.astype(int)
            image_render_d = self.image_render_d[:, bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2], :]
            image_render_d = torch.squeeze(image_render_d)

            depth_mask = image_render_d != -1.0
            loss_dep = -torch.sum((torch.abs(image_render_d[depth_mask] - self.image_depth[depth_mask])).reshape(self.batch_size, -1), -1) / 10000.0

            # print("loss_d = ", loss_dep)
            loss_dep = torch.tensor(0.0)

            # debug
            pred_rgb = self.image_render_rgb.cpu().data.numpy()[0]
            pred_d = image_render_d.cpu().data.numpy()
            gt_d = self.image_depth.clone().detach().cpu().data.numpy()

            cv2.imshow("pred_rgb", pred_rgb)
            cv2.imshow("pred", pred_d)
            cv2.imshow("gt_d", gt_d)
            cv2.waitKey(1)
        else:
            loss_dep = torch.tensor(0.0)




        return loss_seg, loss_2d, loss_reg, loss_dep, self.image_render_d, self.image_render_rgb, hand_joints

