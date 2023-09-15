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
    lr_scale = []

    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if value.requires_grad:
            if 'xy_root' in key:
                lr_xyz_root.append(value)
            elif 'z_root' in key:
                lr_xyz_root.append(value)
            elif 'input_scale' in key:
                lr_scale.append(value)
            elif 'input_rot' in key:
                lr_rot.append(value)
            elif 'input_shape' in key:
                lr_shape.append(value)
            else:       # input_pose
                lr_pose.append(value)

    model_params = [{'params': lr_xyz_root, 'lr': init_lr * 5e0},
                    {'params': lr_rot, 'lr': init_lr},
                    {'params': lr_pose, 'lr': init_lr},
                    {'params': lr_shape, 'lr': init_lr * 2e0},
                    {'params': lr_scale, 'lr': init_lr * 2e0}]
    return model_params

def set_lr_forObj(model, init_lr):
    lr_pose = []

    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        lr_pose.append(value)

    model_params = [{'params': lr_pose, 'lr': init_lr}]
    return model_params


def initialize_optimizer(model, model_obj, lr_init, CFG_WITH_OBJ, lr_init_obj):

    params_hand = set_lr_forHand(model, lr_init)
    if not CFG_WITH_OBJ:
        optimizer = torch.optim.Adam(params_hand)
    else:
        params_obj = set_lr_forObj(model_obj, lr_init_obj)
        params = list(params_hand) + list(params_obj)
        optimizer = torch.optim.Adam(params)

    return optimizer

def update_optimizer(optimizer, ratio_root=1.0, ratio_rot=1.0, ratio_pose=1.0, ratio_shape=1.0, ratio_scale=1.0):
    # order : ': lr_xyz_root, lr_rot, lr_pose, lr_shape, lr_scale

    optimizer.param_groups[0]['lr'] *= ratio_root
    optimizer.param_groups[1]['lr'] *= ratio_rot
    optimizer.param_groups[2]['lr'] *= ratio_pose
    optimizer.param_groups[3]['lr'] *= ratio_shape
    optimizer.param_groups[4]['lr'] *= ratio_scale

    return optimizer




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