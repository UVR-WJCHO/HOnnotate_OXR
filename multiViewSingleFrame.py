import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import torch.nn as nn
import numpy as np

from dataloader import DataLoader
from renderer import Renderer
from handModel import HandModel

from absl import flags
from absl import app

## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('date', '230612', 'target db Name')   ## name ,default, help
flags.DEFINE_string('type', 'bare', 'Sequence Name')
flags.DEFINE_boolean('multiView', False, 'multiView')
FLAGS(sys.argv)

'''
TODO
get config from file(yaml, json, etc.)
'''
## Config
CFG_DATA_DIR = os.path.join(os.getcwd(), 'dataset')
CFG_LR_INIT = 0.2
CFG_NUM_ITER = 100
CFG_DEVICE = 'cuda'
CFG_BATCH_SIZE = 1
CFG_MANO_PATH = os.path.join(os.getcwd(), 'models')



def main(argv):
    ## Load data of each camera
    mas_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.date, FLAGS.type, 'mas')
    if FLAGS.multiView:
        sub1_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.date, FLAGS.type, 'sub1')
        sub2_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.date, FLAGS.type, 'sub2')
        sub3_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.date, FLAGS.type, 'sub3')

    ## Initialize remderer
    mas_renderer = Renderer(device=CFG_DEVICE, bs=CFG_BATCH_SIZE)
    mas_K, mas_M, mas_D = mas_dataloader.get_cam_parameters()
    mas_renderer = mas_renderer.set_renderer(Ext = mas_M, intrinsics = mas_K)
    if FLAGS.multiView:
        sub1_renderer = Renderer(device=CFG_DEVICE, bs=CFG_BATCH_SIZE)
        sub1_K, sub1_M, sub1_D = sub1_dataloader.get_cam_parameters()
        sub1_renderer = sub1_renderer.set_renderer(Ext = sub1_M, intrinsics = sub1_K)
        sub2_renderer = Renderer(device=CFG_DEVICE, bs=CFG_BATCH_SIZE)
        sub2_K, sub2_M, sub2_D = sub2_dataloader.get_cam_parameters()
        sub2_renderer = sub2_renderer.set_renderer(Ext = sub2_M, intrinsics = sub2_K)
        sub3_renderer = Renderer(device=CFG_DEVICE, bs=CFG_BATCH_SIZE)
        sub3_K, sub3_M, sub3_D = sub3_dataloader.get_cam_parameters()
        sub3_renderer = sub3_renderer.set_renderer(Ext = sub3_M, intrinsics = sub3_K)

    ## Initialize loss function
    loss_func = MultiViewLossFunc()

    if FLAGS.multiView:
        if (len(mas_dataloader) != len(sub1_dataloader)) or (len(mas_dataloader) != len(sub2_dataloader)) or (len(mas_dataloader) != len(sub3_dataloader)):
            raise ValueError("The number of data is not same between cameras")
        
    for frame in range(len(mas_dataloader)):
        mas_sample = mas_dataloader[frame]
        if FLAGS.multiView:
            sub1_sample = sub1_dataloader[frame]
            sub2_sample = sub2_dataloader[frame]
            sub3_sample = sub3_dataloader[frame]
        ## Initialize hand model
        model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE)
        model_params = model.parameters()

        ## Initialize optimizer
        optimizer = torch.optim.Adam(model_params, lr=CFG_LR_INIT)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        for iter in range(CFG_NUM_ITER):
            hand_param = model()
            total_loss = 


        
    
