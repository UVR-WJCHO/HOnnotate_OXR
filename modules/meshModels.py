import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../'))
import torch
import torch.nn as nn
import numpy as np
from manopth.manolayer import ManoLayer

from utils.modelUtils import *
import time

class HandModel(nn.Module):
    def __init__(self, mano_path, device, batch_size, initial_rot=torch.zeros(3), initial_pose=torch.zeros(45), initial_shape=torch.zeros(1, 10), side="right"):
        super(HandModel, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.wrist_idx = 0
        self.mcp_idx = 9
        self.key_bone_len = 10.0
        self.mano_layer = ManoLayer(side=side, mano_root=mano_path, use_pca=False, flat_hand_mean=False,
                                    center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

        # initial pose parameters
        # self.xy_root = nn.Parameter(torch.tensor([-10, 4], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        # self.z_root = nn.Parameter(torch.tensor([50], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        # initial_pose = torch.tensor([[0.0211, -0.3187, 0.0651, 0.0000, 0.0000, -0.6944, 0.0000, 0.0000,
        #                            0.2245, -0.0072, -0.2008, 0.2014, 0.0000, 0.0000, -0.7271, 0.0000,
        #                            0.0000, 0.2305, -0.0091, -0.1125, -0.1157, 0.0883, 0.0000, -0.5734,
        #                            0.0000, 0.0000, 0.1881, -0.0093, -0.3019, 0.0766, 0.0000, 0.0000,
        #                            -0.6934, 0.0000, 0.0000, 0.3669, -0.5000, -0.2614, 0.2612, 0.0000,
        #                            -0.5810, 0.0000, 0.0000, 0.8677, -0.5000]])
        # initial_rot = torch.tensor([[-0.9538, -1.6024, 1.4709]])

        self.xy_root = nn.Parameter(
            torch.tensor([-0.9094, 12.0501], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        self.z_root = nn.Parameter(torch.tensor([65.0], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        initial_pose = torch.tensor([[-0.1072, 0.1475, -0.7427, -0.1397, 0.1483, -0.0789, -0.0184, 0.0225,
                 7.2649, -0.1668, 0.8281, 0.7914, -0.0104, 0.1238, 2.4139, 0.0329,
                 0.0267, 2.3347, -0.0112, 0.4245, 0.5857, 0.1500, 0.0563, 1.5343,
                 -0.0234, -0.0344, 3.6133, -0.0661, 0.6330, 0.5967, -0.0478, 0.0238,
                 2.9334, 0.0396, -0.0266, 2.5399, -0.2902, 0.2651, 0.1446, 0.1516,
                 3.0003, -0.0806, 0.0826, 3.0492, -0.4458]])
        initial_rot = torch.tensor([[-1.7276, -1.6758, 2.1557]])

        self.input_rot = nn.Parameter(clip_mano_hand_rot(initial_rot.to(device)))
        self.input_pose = nn.Parameter(initial_pose.to(device))
        self.input_shape = nn.Parameter(initial_shape.repeat(self.batch_size, 1).to(device))

        # Inital set up
        self.pose_all = torch.cat((self.input_rot, self.input_pose), 1)
        # normalize scale
        hand_verts, hand_joints = self.mano_layer(self.pose_all, self.input_shape)
        scale = torch.tensor([[self.compute_normalized_scale(hand_joints)]])
        self.input_scale = nn.Parameter(scale.repeat(self.batch_size, 1).to(device))


    def compute_normalized_scale(self, hand_joints):
        return (torch.sum((hand_joints[0, self.mcp_idx] - hand_joints[0, self.wrist_idx])**2)**0.5)/self.key_bone_len

    def change_grads(self, root=False, rot=False, pose=False, shape=False, scale=False):
        self.xy_root.requires_grad = root
        self.z_root.requires_grad = root
        self.input_rot.requires_grad = rot
        self.input_pose.requires_grad = pose
        self.input_shape.requires_grad = shape
        self.input_scale.requires_grad = scale

    def forward(self):
        self.pose_ = self.input_pose
        self.shape_ = self.input_shape

        # need clipping?
        # self.pose_.data = clip_mano_hand_pose(self.input_pose)
        self.pose_.data = self.input_pose
        mano_param = torch.cat([self.input_rot, self.pose_], dim=1)

        hand_verts, hand_joints = self.mano_layer(mano_param, self.shape_)

        xyz_root = torch.cat([self.xy_root, self.z_root], dim=-1)
        hand_verts = hand_verts / self.input_scale
        hand_verts = hand_verts + xyz_root[:, None, :]
        hand_joints = hand_joints / self.input_scale
        hand_joints = hand_joints + xyz_root[:, None, :]

        hand_faces = self.mano_layer.th_faces.repeat(self.batch_size, 1, 1)

        return {'verts':hand_verts, 'faces':hand_faces, 'joints':hand_joints, 'xyz_root':xyz_root, 'scale':self.input_scale, 'rot':self.input_rot, 'pose':self.pose_, 'shape':self.shape_}


class ObjModel(nn.Module):
    def __init__(self, device, batch_size, obj_template, obj_init_pose=np.eye(4)):
        super(ObjModel, self).__init__()

        self.device = device
        self.batch_size = batch_size

        template_verts, template_faces = obj_template['verts'], obj_template['faces']
        # template_faces = np.array([np.array(x) for x in template_faces])
        # self.template_faces = torch.unsqueeze(torch.FloatTensor(template_faces), axis=0).to(self.device)
        # template_verts = np.asarray(template_verts)[:, :3]
        # self.template_verts = torch.FloatTensor(template_verts).to(self.device)

        self.template_faces = template_faces.to(self.device).unsqueeze(0)
        self.template_verts = template_verts.to(self.device)

        # only obj_pose is trainable [bs, 3, 4]
        obj_pose = torch.tensor(obj_init_pose, dtype=torch.float32)
        obj_pose = obj_pose.view(self.batch_size, -1)
        self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_pose.requires_grad = True

        self.h = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).to(self.device)

    def get_object_mat(self):
        obj_mat = torch.cat([self.obj_pose, self.h], dim=-1)
        return np.squeeze(obj_mat.detach().cpu().numpy())

    def update_pose(self, pose):
        obj_pose = torch.tensor(pose, dtype=torch.float32)
        obj_pose = obj_pose.view(self.batch_size, -1)
        self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_pose.requires_grad = True

    def apply_transform(self, obj_pose, obj_verts):
        obj_pose = obj_pose.view(3, 4)
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

        # debug_obj_pose = self.obj_pose.view(4, 4).detach().cpu().numpy()
        # debug_template_verts = np.squeeze(self.template_verts.detach().cpu().numpy())
        #
        # homogeneous_points = np.hstack((debug_template_verts, np.ones((debug_template_verts.shape[0], 1))))
        #
        # # Apply matrix multiplication
        # transformed_points = np.dot(debug_obj_pose, homogeneous_points.T).T
        #
        # # Convert back to Cartesian coordinates
        # transformed_points_cartesian = transformed_points[:, :3] / transformed_points[:, 3:]

        obj_verts = self.apply_transform(self.obj_pose, self.template_verts)

        return {'verts': obj_verts, 'faces': self.template_faces, 'pose': self.obj_pose}

