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
        self.mano_layer = ManoLayer(side=side, mano_root=mano_path, use_pca=False, flat_hand_mean=True,
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
        # initial_pose = torch.tensor([[-1.2770e-02,  3.5734e-02,  1.0304e-01, -1.2469e-02,  1.7843e-02,
        #           1.3339e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.6089e-03,
        #           8.1811e-01,  6.0679e-01,  7.8873e-03,  5.2546e-03,  1.0138e+00,
        #           0.0000e+00,  0.0000e+00,  0.0000e+00, -3.1368e-02,  1.5023e+00,
        #           9.1416e-01,  9.1727e-02,  4.5114e-03, -1.1004e-01,  0.0000e+00,
        #           0.0000e+00,  0.0000e+00, -1.4536e-02,  1.0635e+00,  1.5662e+00,
        #           1.7692e-02,  4.9017e-03, -7.8507e-01,  0.0000e+00,  0.0000e+00,
        #           0.0000e+00, -5.0363e-01, -4.4525e-01,  2.8814e-01,  1.1674e-02,
        #          -7.1911e-01, -1.3209e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
        initial_rot = torch.tensor([[-1.7276, -1.6758, 2.1557]])
        initial_pose = torch.zeros((1, 45), dtype=torch.float32)

        ## grab last tip
        # initial_pose[0, 43] = torch.tensor(-1.5)    # thumb
        # initial_pose[0, 35] = torch.tensor(1.5)  # ring
        initial_pose[0, 26] = torch.tensor(1.5)  # pinky
        # initial_pose[0, 17] = torch.tensor(1.5)  # middle
        # initial_pose[0, 8] = torch.tensor(1.5)  # index
        #
        # initial_pose[0, 40] = torch.tensor(-1)  # thumb
        # initial_pose[0, 32] = torch.tensor(1.)  # ring
        initial_pose[0, 23] = torch.tensor(1.)  # pinky
        # initial_pose[0, 14] = torch.tensor(1.)  # middle
        # initial_pose[0, 5] = torch.tensor(1.)  # index
        #
        # initial_pose[0, 24] = torch.tensor(1.)  # pinky

        # initial_pose = torch.zeros((1, 30), dtype=torch.float32)
        # initial_tip_pose = torch.zeros((1, 15), dtype=torch.float32)

        self.input_rot = nn.Parameter(clip_mano_hand_rot(initial_rot.to(device)))
        self.input_pose = nn.Parameter(initial_pose.to(device))
        # self.input_tip_pose = nn.Parameter(initial_tip_pose.to(device))

        self.input_shape = nn.Parameter(initial_shape.repeat(self.batch_size, 1).to(device))

        # Inital set up
        # pose_all = self.compute_pose_all(self.input_pose, self.input_tip_pose)  # (1, 45)
        pose_all = self.input_pose
        self.pose_all = torch.cat((self.input_rot, pose_all), 1)
        # normalize scale
        hand_verts, hand_joints = self.mano_layer(self.pose_all, self.input_shape)
        scale = torch.tensor([[self.compute_normalized_scale(hand_joints)]])
        self.input_scale = nn.Parameter(scale.repeat(self.batch_size, 1).to(device))


    def compute_pose_all(self, pose, tip_pose):
        # pose : (1, 30)
        # tip_pose : (1, 15)
        pose_all = torch.cat((pose[:, :6], tip_pose[:, :3],
                              pose[:, 6:12], tip_pose[:, 3:6],
                              pose[:, 12:18], tip_pose[:, 6:9],
                              pose[:, 18:24], tip_pose[:, 9:12],
                              pose[:, 24:30], tip_pose[:, 12:15]), 1)
        return pose_all

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
        self.shape_ = self.input_shape

        self.pose_ = self.input_pose

        # need clipping?
        # self.pose_.data = clip_mano_hand_pose(self.input_pose)
        # self.pose_.data = self.input_pose
        # self.pose_ = self.compute_pose_all(self.input_pose, self.input_tip_pose)

        mano_param = torch.cat([self.input_rot, self.pose_], dim=1)

        hand_verts, hand_joints = self.mano_layer(mano_param, self.shape_)

        xyz_root = torch.cat([self.xy_root, self.z_root], dim=-1)
        hand_verts = hand_verts / self.input_scale
        hand_verts = hand_verts + xyz_root[:, None, :]
        hand_joints = hand_joints / self.input_scale
        hand_joints = hand_joints + xyz_root[:, None, :]

        hand_faces = self.mano_layer.th_faces.repeat(self.batch_size, 1, 1)

        return {'verts':hand_verts, 'faces':hand_faces, 'joints':hand_joints, 'xyz_root':xyz_root,
                'scale':self.input_scale, 'rot':self.input_rot, 'pose':self.pose_, 'shape':self.shape_}


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

