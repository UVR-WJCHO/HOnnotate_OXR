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
        self.xy_root = nn.Parameter(torch.tensor([-10, 4], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        self.z_root = nn.Parameter(torch.tensor([50], dtype=torch.float32).repeat(self.batch_size, 1).to(device))

        initial_pose = torch.tensor([[0.0211, -0.3187, 0.0651, 0.0000, 0.0000, -0.6944, 0.0000, 0.0000,
                                   0.2245, -0.0072, -0.2008, 0.2014, 0.0000, 0.0000, -0.7271, 0.0000,
                                   0.0000, 0.2305, -0.0091, -0.1125, -0.1157, 0.0883, 0.0000, -0.5734,
                                   0.0000, 0.0000, 0.1881, -0.0093, -0.3019, 0.0766, 0.0000, 0.0000,
                                   -0.6934, 0.0000, 0.0000, 0.3669, -0.5000, -0.2614, 0.2612, 0.0000,
                                   -0.5810, 0.0000, 0.0000, 0.8677, -0.5000]])
        initial_rot = torch.tensor([[-0.9538, -1.6024, 1.4709]])

        self.input_rot = nn.Parameter(clip_mano_hand_rot(initial_rot.to(device)))
        self.input_pose = nn.Parameter(initial_pose.to(device))
        self.input_shape = nn.Parameter(initial_shape.repeat(self.batch_size, 1).to(device))

        # Inital set up
        self.pose_adjusted_all = torch.cat((self.input_rot, self.input_pose), 1)
        # normalize scale
        hand_verts, hand_joints = self.mano_layer(self.pose_adjusted_all, self.input_shape)
        self.scale = torch.tensor([[self.compute_normalized_scale(hand_joints)]]).to(device)


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
        self.pose_.data = clip_mano_hand_pose(self.input_pose)
        self.shape_.data = clip_mano_hand_shape(self.input_shape)
        mano_param = torch.cat([self.input_rot, self.pose_], dim=1)

        hand_verts, hand_joints = self.mano_layer(mano_param, self.shape_)
        hand_faces = self.mano_layer.th_faces.repeat(self.batch_size, 1, 1)

        xyz_root = torch.cat([self.xy_root, self.z_root], dim=-1)
        hand_verts = hand_verts / self.scale
        hand_verts = hand_verts + xyz_root[:, None, :]
        hand_joints = hand_joints / self.scale
        hand_joints = hand_joints + xyz_root[:, None, :]

        return {'verts':hand_verts, 'faces':hand_faces, 'joints':hand_joints, 'xyz_root':xyz_root, 'scale':self.scale, 'rot':self.input_rot, 'pose':self.pose_, 'shape':self.shape_}

