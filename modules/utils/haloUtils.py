import torch
import numpy as np

from dependencies.halo.halo.models.halo_adapter.converter import PoseConverter, transform_to_canonical
from dependencies.halo.halo.models.halo_adapter.interface import (get_halo_model, convert_joints, change_axes, scale_halo_trans_mat)
from dependencies.halo.halo.models.halo_adapter.projection import get_projection_layer
from dependencies.halo.halo.models.halo_adapter.transform_utils import xyz_to_xyz1


def get_bone_lengths(joints):
    bones = np.array([
        (0,4),
        (1,2),
        (2,3),
        (3,17),
        (4,5),
        (5,6),
        (6,18),
        (7,8),
        (8,9),
        (9,20),
        (10,11),
        (11,12),
        (12,19),
        (13,14),
        (14,15),
        (15,16)
    ])

    bone_length = joints[bones[:,0]] - joints[bones[:,1]]
    bone_length = np.linalg.norm(bone_length, axis=1)

    return bone_length


# Codes adopted from HALO
def preprocess_joints(joints, side='right', scale=0.4):

    permute_mat = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]

    joints = joints[0]
    joints = joints[permute_mat]

    if side == 'left':
        joints *= [-1, 1, 1]

    org_joints = joints

    joints = joints.unsqueeze(0) #torch.Tensor(joints).unsqueeze(0)
    joints = convert_joints(joints, source='halo', target='biomech')
    is_right_vec = torch.ones(joints.shape[0])

    pose_converter = PoseConverter()

    palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(joints, is_right=is_right_vec)
    palm_align_kps_local_cs_nasa_axes, swap_axes_mat = change_axes(palm_align_kps_local_cs)

    rot_then_swap_mat = torch.matmul(swap_axes_mat.unsqueeze(0), glo_rot_right.float()).unsqueeze(0)

    trans_mat_pc, _ = pose_converter(palm_align_kps_local_cs_nasa_axes, is_right_vec.cuda())
    trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='halo')

    joints_for_nasa_input = [0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    trans_mat_pc = trans_mat_pc[:, joints_for_nasa_input]

    org_joints = torch.matmul(rot_then_swap_mat.squeeze(), xyz_to_xyz1(org_joints).unsqueeze(-1))[:, :3, 0]
    bone_lengths = torch.Tensor(get_bone_lengths(org_joints.cpu().numpy())).squeeze()

    trans_mat_pc_all = trans_mat_pc
    unpose_mat = scale_halo_trans_mat(trans_mat_pc_all)

    scale_mat = torch.eye(4).cuda() * scale
    scale_mat[3, 3] = 1.
    unpose_mat = torch.matmul(unpose_mat, scale_mat).squeeze()

    return unpose_mat, torch.Tensor(bone_lengths).squeeze(0), rot_then_swap_mat.squeeze()

