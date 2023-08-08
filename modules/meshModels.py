import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../'))
import torch
import torch.nn as nn
import numpy as np
from manopth.manolayer import ManoLayer

from utils.modelUtils import *


class HandModel(nn.Module):
    def __init__(self, mano_path, device, batch_size, initial_rot=torch.zeros(3), initial_pose=torch.zeros(45), initial_shape=torch.zeros(1, 10)):
        super(HandModel, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.wrist_idx = 0
        self.mcp_idx = 9
        self.key_bone_len = 10.0
        self.mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=False,
                                    center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

        # initial pose parameters
        self.xy_root = nn.Parameter(torch.tensor([0.6595, 1.0659], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        self.z_root = nn.Parameter(torch.tensor([56], dtype=torch.float32).repeat(self.batch_size, 1).to(device))

        initial_pose = torch.tensor([[1.5464e-02, -1.5886e-02, -2.6480e-01, 1.7313e-03, -3.0119e-03,
                 -4.3855e-01, 8.4188e-06, -1.5501e-04, 6.5009e+00, -2.0439e-02,
                 4.3690e-02, -4.9303e-01, -1.1642e-03, 1.1375e-03, -2.8073e-01,
                 4.0919e-05, 2.0615e-03, 6.1110e-02, -2.3347e-02, -7.9014e-02,
                 -4.5420e-01, 8.7698e-02, -4.7082e-03, -4.4077e-01, 1.5274e-03,
                 -2.9957e-03, 1.4533e-01, -2.2924e-02, -1.6601e-01, -3.8383e-01,
                 1.3930e-03, -1.4208e-03, -5.2633e-01, 2.6763e-04, -2.0722e-03,
                 2.8310e-01, -4.7774e-01, 6.2213e-03, 3.3319e-01, 3.3308e-02,
                 -6.8497e-01, 4.3174e-02, 2.7197e-02, 8.1599e-01, -4.7765e-01]])
        initial_rot = torch.tensor([[-1.3630, -1.8802, 1.8825]])

        self.input_rot = nn.Parameter(clip_mano_hand_rot(initial_rot.to(device)))
        self.input_pose = nn.Parameter(initial_pose.to(device))
        self.input_shape = nn.Parameter(initial_shape.repeat(self.batch_size, 1).to(device))

        # Inital set up
        self.pose_all = torch.cat((self.input_rot, self.input_pose), 1)
        # normalize scale
        hand_verts, hand_joints = self.mano_layer(self.pose_all, self.input_shape)
        self.scale = nn.Parameter(torch.tensor([[self.compute_normalized_scale(hand_joints)]])).to(device)


    def compute_normalized_scale(self, hand_joints):
        return (torch.sum((hand_joints[0, self.mcp_idx] - hand_joints[0, self.wrist_idx])**2)**0.5)/self.key_bone_len

    def change_grads(self, root=False, rot=False, pose=False, shape=False):
        self.xy_root.requires_grad = root
        self.z_root.requires_grad = root
        self.input_rot.requires_grad = rot
        self.input_pose.requires_grad = pose
        self.input_shape.requires_grad = shape
        # self.input_scale.requires_grad = scale

    def forward(self):
        self.pose_ = self.input_pose
        self.shape_ = self.input_shape

        # need clipping?
        # self.pose_.data = clip_mano_hand_pose(self.input_pose)
        self.pose_.data = self.input_pose
        mano_param = torch.cat([self.input_rot, self.pose_], dim=1)

        hand_verts, hand_joints = self.mano_layer(mano_param, self.shape_)

        xyz_root = torch.cat([self.xy_root, self.z_root], dim=-1)
        hand_verts = hand_verts / self.scale
        hand_verts = hand_verts + xyz_root[:, None, :]
        hand_joints = hand_joints / self.scale
        hand_joints = hand_joints + xyz_root[:, None, :]

        hand_faces = self.mano_layer.th_faces.repeat(self.batch_size, 1, 1)

        return {'verts':hand_verts, 'faces':hand_faces, 'joints':hand_joints, 'xyz_root':xyz_root, 'scale':self.scale, 'rot':self.input_rot, 'pose':self.pose_, 'shape':self.shape_}


class ObjModel(nn.Module):
    def __init__(self, device, batch_size, obj_template, obj_init_pose=np.eye(4)):
        super(ObjModel, self).__init__()

        self.device = device
        self.batch_size = batch_size

        template_verts,  template_faces = obj_template['verts'], obj_template['faces']
        template_faces = np.asarray(template_faces)
        self.template_faces = torch.unsqueeze(torch.FloatTensor(template_faces), axis=0).to(self.device)

        template_verts = np.asarray(template_verts)[:, :3]
        self.template_verts = torch.FloatTensor(template_verts).to(self.device)

        # only obj_pose is trainable [bs, 4, 4]     ... currently consider only 1 batch
        # trainable obj pose = (3, 4)
        obj_pose = torch.tensor(obj_init_pose, dtype=torch.float32)[:-1, :]
        obj_pose = obj_pose.view(self.batch_size, -1)
        self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_pose.requires_grad = True


        self.h = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).to(self.device)

        self.scale = torch.FloatTensor([100.0]).to(self.device)


    def update_pose(self, pose):
        obj_pose = torch.tensor(pose, dtype=torch.float32)[:-1, :]
        obj_pose = obj_pose.view(self.batch_size, -1)
        self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_pose.requires_grad = True

    def apply_transform(self, obj_pose, obj_verts):
        obj_pose = obj_pose.view(3, 4)
        obj_mat = torch.cat([obj_pose, self.h], dim=0)

        # Append 1 to each coordinate to convert them to homogeneous coordinates
        h = torch.ones((obj_verts.shape[0], 1)).to(self.device)
        homogeneous_points = torch.concatenate((obj_verts, h), 1)

        # Apply matrix multiplication
        transformed_points = homogeneous_points @ obj_mat.T
        # Convert back to Cartesian coordinates
        transformed_points_cartesian = transformed_points[:, :3] / transformed_points[:, 3:]
        transformed_points_cartesian = transformed_points_cartesian.view(self.batch_size, -1, 3)

        return transformed_points_cartesian * self.scale

    def forward(self):
        obj_verts = self.apply_transform(self.obj_pose, self.template_verts)

        return {'verts':obj_verts, 'faces':self.template_faces, 'pose':self.obj_pose}

