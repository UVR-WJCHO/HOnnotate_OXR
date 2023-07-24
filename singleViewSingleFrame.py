import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import torch.nn as nn
import numpy as np

from dataloader import DataLoader
from renderer import Renderer
from handModel import HandModel
from lossFunc import SingleViewLossFunc
from utils import *
from utils import params

from absl import flags
from absl import app
from absl import logging

## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('date', '230612', 'target db Name')   ## name ,default, help
flags.DEFINE_string('type', 'bare', 'Sequence Name')
flags.DEFINE_string('cam', 'mas', 'main target camera')
FLAGS(sys.argv)

'''
TODO
get config from file(yaml, json, etc.)
'''
## Config
CFG_DATA_DIR = os.path.join(os.getcwd(), 'dataset')
CFG_LR_INIT = 0.4
CFG_NUM_ITER = 500
CFG_DEVICE = 'cuda'
CFG_BATCH_SIZE = 1
CFG_MANO_PATH = os.path.join(os.getcwd(), 'models')
CFG_LOSS_DICT = ['kpts2d', 'reg']
CFG_IMAGE_SIZE = (1080, 1920) # (H, W)
CFG_SAVE_PATH = os.path.join(os.getcwd(), 'output')

torch.autograd.set_detect_anomaly(True)
def main(argv):
    ## Load data of each camera
    cam_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.date, FLAGS.type, FLAGS.cam)

    ## Initialize remderer
    cam_renderer = Renderer(device=CFG_DEVICE, bs=CFG_BATCH_SIZE)
    cam_params = cam_dataloader.get_cam_parameters()
    cam_renderer.set_renderer(Ext = cam_params["Ms"], intrinsics = cam_params["Ks"], image_size = CFG_IMAGE_SIZE)

    ## Initialize loss function
    loss_func = SingleViewLossFunc(device=CFG_DEVICE, bs=CFG_BATCH_SIZE)
      
    for frame in range(len(cam_dataloader)):
        cam_sample = cam_dataloader[frame]
        loss_func.set_gt(cam_sample, cam_params, cam_renderer, CFG_LOSS_DICT)
        ## Initialize hand model
        model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE, initial_rot=params.initial_rot, initial_pose=params.initial_pose)
        # model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE)
        model_params = model.parameters()

        ## Initialize optimizer
        optimizer = torch.optim.Adam(model_params, lr=CFG_LR_INIT)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        for iter in range(CFG_NUM_ITER):
            target_hand = model()
            """
            {'verts':hand_verts, 'faces':hand_faces, 'joints':hand_joints, 'xyz_root':xyz_root, 'scale':self.input_scale, 'rot':self.input_rot, 'pose':self.input_pose, 'shape':self.input_shape}
            hand vertices, hand joints in mano coordinates
            """
            losses = loss_func(target_hand)

            total_loss = sum(losses[k] for k in CFG_LOSS_DICT)

            
            logs = ["Iter: {}, Loss: {}".format(iter, total_loss)]
            logs += ['[%s:%.4f]'%(key, losses[key]) for key in losses.keys() if key in CFG_LOSS_DICT]
            logging.info(''.join(logs))

            loss_func.visualize(CFG_SAVE_PATH)
            
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

        # Save [frame] result
        # target_hand = model()
        """
        TODO
        save result
        """

        break

if __name__ == "__main__":
    app.run(main)