import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import torch.nn as nn
import numpy as np
import cv2

from modules.dataloader import DataLoader
from modules.renderer import Renderer
from modules.handModel import HandModel
from modules.lossFunc import MultiViewLossFunc
from utils import *
from utils import params

from absl import flags
from absl import app
from absl import logging


## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230612', 'target db name')   ## name ,default, help
flags.DEFINE_string('type', 'banana', 'target sequence name')
FLAGS(sys.argv)

'''
TODO
get config from file(yaml, json, etc.)
'''
## Config
CFG_DATA_DIR = os.path.join(os.getcwd(), 'dataset')
CFG_LR_INIT = 0.4
CFG_NUM_ITER = 100
CFG_DEVICE = 'cuda'
CFG_BATCH_SIZE = 1
CFG_MANO_PATH = os.path.join(os.getcwd(), 'modules', 'mano', 'models')

CFG_LOSS_DICT = ['kpts2d', 'reg']#, 'depth', 'seg']
CFG_SAVE_PATH = os.path.join(os.getcwd(), 'output')
CFG_CAMID_SET = ['mas', 'sub1', 'sub2', 'sub3']

CFG_IMG_WIDTH = 1920
CFG_IMG_HEIGHT = 1080
CFG_CROP_IMG_WIDTH = 640
CFG_CROP_IMG_HEIGHT = 480

def main(argv):
    ## Load data of each camera
    mas_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'mas')
    sub1_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'sub1')
    sub2_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'sub2')
    sub3_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'sub3')

    ## Initialize renderer, every renderer's extrinsic is set to master camera extrinsic
    mas_K, mas_M, mas_D = mas_dataloader.cam_parameter
    mas_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, mas_M, mas_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
    sub1_K, _, sub1_D = sub1_dataloader.cam_parameter
    sub1_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, mas_M, sub1_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
    sub2_K, _, sub2_D = sub2_dataloader.cam_parameter
    sub2_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, mas_M, sub2_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
    sub3_K, _, sub3_D = sub3_dataloader.cam_parameter
    sub3_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, mas_M, sub3_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))

    dataloader_set = [mas_dataloader, sub1_dataloader, sub2_dataloader, sub3_dataloader]
    renderer_set = [mas_renderer, sub1_renderer, sub2_renderer, sub3_renderer]

    ## Initialize loss function
    loss_func = MultiViewLossFunc(device=CFG_DEVICE)

    if (len(mas_dataloader) != len(sub1_dataloader)) or (len(mas_dataloader) != len(sub2_dataloader)) or (len(mas_dataloader) != len(sub3_dataloader)):
        raise ValueError("The number of data is not same between cameras")

    ## Initialize hand model
    model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE)
    model.change_grads(root=True, rot=True, pose=True, shape=False)
    model_params = model.parameters()

    flag_render = False
    if 'depth' in CFG_LOSS_DICT or 'seg' in CFG_LOSS_DICT:
        flag_render = True

    for frame in range(len(mas_dataloader)):
        # frame 0 is often errorneous
        if frame == 0:
            continue

        ## Initialize optimizer
        lr_init = CFG_LR_INIT
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
        # skipped shape paramter lr
        optimizer = torch.optim.Adam(model_params, lr=CFG_LR_INIT)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        ## Set main cam of calibration process
        main_cam_params = mas_dataloader.cam_parameter

        for iter in range(CFG_NUM_ITER):
            loss_all = {'kpts2d':0.0, 'depth':0.0, 'seg':0.0, 'reg':0.0}
            for camIdx, camID in enumerate(CFG_CAMID_SET):
                # if not camIdx == 1:
                #     continue
                cam_params = dataloader_set[camIdx].cam_parameter
                cam_sample = dataloader_set[camIdx][frame]
                # skip non-detected camera
                if np.isnan(cam_sample['kpts3d']).any():
                    continue
                loss_func.set_gt(cam_sample, cam_params, renderer_set[camIdx], CFG_LOSS_DICT, main_cam_params)

                hand_param = model()
                losses = loss_func(hand_param, render=flag_render)
                for k in CFG_LOSS_DICT:
                    loss_all[k] += losses[k]

                # loss_func.visualize(CFG_SAVE_PATH, camID)

            total_loss = sum(loss_all[k] for k in CFG_LOSS_DICT)
            logs = ["Iter: {}, Loss: {}".format(iter, total_loss.data)]
            logs += ['[%s:%.4f]' % (key, loss_all[key]) for key in loss_all.keys() if key in CFG_LOSS_DICT]
            logging.info(''.join(logs))

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

        # visualization results of frame
        for camIdx, camID in enumerate(CFG_CAMID_SET):
            cam_params = dataloader_set[camIdx].cam_parameter
            cam_sample = dataloader_set[camIdx][frame]
            # skip non-detected camera
            if np.isnan(cam_sample['kpts3d']).any():
                continue
            loss_func.set_gt(cam_sample, cam_params, renderer_set[camIdx], CFG_LOSS_DICT, main_cam_params)
            hand_param = model()
            _ = loss_func(hand_param, render=True)
            loss_func.visualize(CFG_SAVE_PATH, camID)

        print("end frame")
        cv2.waitKey(0)


if __name__ == "__main__":
    app.run(main)