import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
from utils import params
from config import *

def set_lr_forHand(model, init_lr):
    lr_xyz_root = []
    lr_rot = []
    lr_pose = []
    lr_shape = []

    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if value.requires_grad:
            if 'xy_root' in key:
                lr_xyz_root.append(value)
            elif 'z_root' in key:
                lr_xyz_root.append(value)
            elif 'input_rot' in key:
                lr_rot.append(value)
            elif 'input_pose' in key:
                lr_pose.append(value)
            elif 'input_shape' in key:
                lr_shape.append(value)

    model_params = [{'params': lr_xyz_root, 'lr': init_lr * 10},
                    {'params': lr_rot, 'lr': init_lr},
                    {'params': lr_pose, 'lr': init_lr},
                    {'params': lr_shape, 'lr': init_lr * 2.0}]
    return model_params

def set_lr_forObj(model, init_lr):
    lr_pose = []

    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        lr_pose.append(value)

    model_params = [{'params': lr_pose, 'lr': init_lr}]
    return model_params


def initialize_optimizer(model):
    lr_xyz_root = []
    lr_rot = []
    lr_pose = []
    lr_shape = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if value.requires_grad:
            if 'xy_root' in key:
                lr_xyz_root.append(value)
            elif 'z_root' in key:
                lr_xyz_root.append(value)
            elif 'input_rot' in key:
                lr_rot.append(value)
            elif 'input_pose' in key:
                lr_pose.append(value)
            elif 'input_shape' in key:
                lr_shape.append(value)

    model_params = [{'params': lr_xyz_root, 'lr': 0.5},
                    {'params': lr_rot, 'lr': 0.05},
                    {'params': lr_pose, 'lr': 0.05}]
    return model_params

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