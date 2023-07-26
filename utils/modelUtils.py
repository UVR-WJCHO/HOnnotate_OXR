import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
from utils import params


def clip_mano_hand_rot(rot_tensor):
    rot_min_tensor = torch.tensor([
        -10.0, -10.0, -10.0
        ]).cuda()
        
    rot_max_tensor = torch.tensor([
        10.0, 10.0, 10.0
        ]).cuda()
    return torch.min(torch.max(rot_tensor, rot_min_tensor), rot_max_tensor)
    
def clip_mano_hand_pose(pose_tensor):
    device = pose_tensor.device
    pose_min_tensor = torch.tensor(params.rot_min_list[3:]).to(device)
    pose_max_tensor = torch.tensor(params.rot_max_list[3:]).to(device)
    return torch.min(torch.max(pose_tensor, pose_min_tensor), pose_max_tensor)
    
def clip_mano_hand_shape(shape_tensor):
    device = shape_tensor.device
    min_val = -10.0
    max_val = 10.0
    shape_min_tensor = torch.tensor([
        min_val, min_val, min_val, min_val, min_val,
        min_val, min_val, min_val, min_val, min_val
    ]).to(device)
    shape_max_tensor = torch.tensor([
        max_val, max_val, max_val, max_val, max_val,
        max_val, max_val, max_val, max_val, max_val
    ]).to(device)
    return torch.min(torch.max(shape_tensor, shape_min_tensor), shape_max_tensor)