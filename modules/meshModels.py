import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../'))
import torch
import torch.nn as nn
import numpy as np
from manopth.manolayer import ManoLayer
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix 

from utils.modelUtils import *
import time

class HandModel(nn.Module):
    def __init__(self, mano_path, device, batch_size, side="right"):
        super(HandModel, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.wrist_idx = 0
        self.mcp_idx = 9
        self.key_bone_len = 10.0
        self.mano_layer = ManoLayer(side=side, mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                                    center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

        initial_shape = torch.zeros(1, 10)
        self.input_shape = nn.Parameter(initial_shape.repeat(self.batch_size, 1).to(device))

        self.xy_root = nn.Parameter(
            torch.tensor([-0.9094, 12.0501], dtype=torch.float32).repeat(self.batch_size, 1).to(device))
        self.z_root = nn.Parameter(torch.tensor([65.0], dtype=torch.float32).repeat(self.batch_size, 1).to(device))

        initial_rot = torch.tensor([[-1.7276, -1.6758, 2.1557]])
        self.input_rot = nn.Parameter(clip_mano_hand_rot(initial_rot.to(device)))

        ## original
        # initial_pose = torch.zeros((1, 45), dtype=torch.float32)
        # self.input_pose = nn.Parameter(initial_pose.to(device))

        ## parts pose
        initial_pose_1 = torch.zeros((1, 15), dtype=torch.float32)
        initial_pose_2 = torch.zeros((1, 15), dtype=torch.float32)
        initial_pose_3 = torch.zeros((1, 15), dtype=torch.float32)

        self.input_pose_1 = nn.Parameter(initial_pose_1.to(device))
        self.input_pose_2 = nn.Parameter(initial_pose_2.to(device))
        self.input_pose_3 = nn.Parameter(initial_pose_3.to(device))


        # Inital set up
        # pose_all = self.input_pose
        pose_all = self.compute_pose_all(self.input_pose_1, self.input_pose_2, self.input_pose_3)  # (1, 45)

        self.pose_all = torch.cat((self.input_rot, pose_all), 1)
        # normalize scale
        hand_verts, hand_joints = self.mano_layer(self.pose_all, self.input_shape)
        scale = torch.tensor([[self.compute_normalized_scale(hand_joints)]])
        # scale = torch.tensor([[8.0]])
        self.input_scale = nn.Parameter(scale.repeat(self.batch_size, 1).to(device))


    def compute_pose_all(self, pose_1, pose_2, pose_3):
        pose_all = torch.cat((pose_1[:, :3], pose_2[:, :3], pose_3[:, :3],
                              pose_1[:, 3:6], pose_2[:, 3:6], pose_3[:, 3:6],
                              pose_1[:, 6:9], pose_2[:, 6:9], pose_3[:, 6:9],
                              pose_1[:, 9:12], pose_2[:, 9:12], pose_3[:, 9:12],
                              pose_1[:, 12:15], pose_2[:, 12:15], pose_3[:, 12:15]), 1)
        return pose_all

    def compute_normalized_scale(self, hand_joints):
        return (torch.sum((hand_joints[0, self.mcp_idx] - hand_joints[0, self.wrist_idx])**2)**0.5)/self.key_bone_len

    # def change_grads(self, root=False, rot=False, pose=False, shape=False, scale=False):
    #     self.xy_root.requires_grad = root
    #     self.z_root.requires_grad = root
    #     self.input_rot.requires_grad = rot
    #     self.input_pose.requires_grad = pose
    #     self.input_shape.requires_grad = shape
    #     self.input_scale.requires_grad = scale

    def change_grads_parts(self, root=False, rot=False, pose_1=False, pose_2=False, pose_3=False, shape=False, scale=False):
        self.xy_root.requires_grad = root
        self.z_root.requires_grad = root
        self.input_rot.requires_grad = rot

        self.input_pose_1.requires_grad = pose_1
        self.input_pose_2.requires_grad = pose_2
        self.input_pose_3.requires_grad = pose_3

        self.input_shape.requires_grad = shape
        self.input_scale.requires_grad = scale

    def change_grads_all(self, root=False, rot=False, pose=False, shape=False, scale=False):
        self.xy_root.requires_grad = root
        self.z_root.requires_grad = root
        self.input_rot.requires_grad = rot

        self.input_pose_1.requires_grad = pose
        self.input_pose_2.requires_grad = pose
        self.input_pose_3.requires_grad = pose

        self.input_shape.requires_grad = shape
        self.input_scale.requires_grad = scale

    def forward(self):
        self.shape_ = self.input_shape

        # self.pose_ = self.input_pose
        self.pose_ = self.compute_pose_all(self.input_pose_1, self.input_pose_2, self.input_pose_3)

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
    def __init__(self, device, batch_size, obj_template, obj_init_rot=np.eye(3)):
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



        # only obj_pose is trainable
        obj_rot = torch.tensor(obj_init_rot).unsqueeze(0)
        obj_rot = matrix_to_axis_angle(obj_rot)

        obj_trans = torch.zeros((1, 3))
        obj_pose = torch.cat((obj_rot, obj_trans), -1)

        self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_pose.requires_grad = True

        self.h = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).to(self.device)

    def get_object_mat(self):
        #obj_mat = torch.cat([self.obj_pose, self.h], dim=-1)
        obj_rot = axis_angle_to_matrix(self.obj_pose[:, :3]).squeeze()
        obj_trans = self.obj_pose[:, 3:].T
        obj_pose = torch.cat((obj_rot, obj_trans), -1)

        obj_mat = torch.cat([obj_pose, self.h], dim=0)

        return np.squeeze(obj_mat.detach().cpu().numpy())

    def update_pose(self, pose, grad=False):
        obj_rot = torch.tensor(pose[:, :3], dtype=torch.float32)
        obj_rot = matrix_to_axis_angle(obj_rot)
        obj_rot = obj_rot.view(self.batch_size, -1)

        obj_trans = torch.tensor(pose[:, 3:], dtype=torch.float32).T
        obj_pose = torch.cat((obj_rot, obj_trans), -1)

        self.obj_pose = nn.Parameter(obj_pose.to(self.device))
        self.obj_pose.requires_grad = grad

    def apply_transform(self, obj_pose, obj_verts):
        obj_rot = axis_angle_to_matrix(obj_pose[:, :3]).squeeze()
        obj_trans = obj_pose[:, 3:].T
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

