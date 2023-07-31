import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import torch.nn as nn
import numpy as np
import cv2

from modules.dataloader import DataLoader, ObjectLoader
from modules.renderer import Renderer
from modules.meshModels import HandModel, ObjModel
from modules.lossFunc import MultiViewLossFunc
from utils import *
from utils.modelUtils import initialize_optimizer

from absl import flags
from absl import app
from absl import logging

from config import *
import time

## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230612', 'target db name')   ## name ,default, help
flags.DEFINE_string('type', 'mustard', 'target sequence name')
FLAGS(sys.argv)


def main(argv):
    flag_render = False
    if 'depth' in CFG_LOSS_DICT or 'seg' in CFG_LOSS_DICT:
        flag_render = True

    ## Load data of each camera, save pkl file for second run.
    print("loading data... %s %s " % (FLAGS.db, FLAGS.type))
    mas_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'mas')
    sub1_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'sub1')
    sub2_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'sub2')
    sub3_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type, 'sub3')

    if CFG_WITH_OBJ:
        obj_dataloader = ObjectLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.type)

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
    loss_func = MultiViewLossFunc(device=CFG_DEVICE, dataloaders=dataloader_set, renderers=renderer_set, losses=CFG_LOSS_DICT)
    loss_func.set_main_cam(main_cam_idx=0)


    if (len(mas_dataloader) != len(sub1_dataloader)) or (len(mas_dataloader) != len(sub2_dataloader)) or (len(mas_dataloader) != len(sub3_dataloader)):
        raise ValueError("The number of data is not same between cameras")

    ## Initialize hand model
    model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE)
    model.change_grads(root=True, rot=True, pose=True, shape=True)
    # TODO : duplicated define. check below
    model_params = model.parameters()

    if CFG_WITH_OBJ:
        obj_init_pose = obj_dataloader[0]   # numpy (4, 4) format
        obj_template_mesh = obj_dataloader.obj_mesh_data
        model_obj = ObjModel(CFG_DEVICE, CFG_BATCH_SIZE, obj_template_mesh, obj_init_pose)
        ## Set object's main camera extrinsic
        obj_main_cam_idx = CFG_CAMID_SET.index(obj_dataloader.obj_view)
        loss_func.set_object_main_extrinsic(obj_main_cam_idx)

    for frame in range(len(mas_dataloader)):
        ## initial frames are often errorneous (banana sub2 0000~0002 is error, ...)
        if frame < 5:
            continue

        ## Set object ICG pose as init pose on every frame? or use previous pose?
        if CFG_WITH_OBJ:
            obj_pose = obj_dataloader[frame]
            model_obj.update_pose(pose=obj_pose)

        ## Initialize optimizer
        # TODO : skipped shape lr in model_params. check lr ratio with above definition
        model_params = initialize_optimizer(model)
        optimizer = torch.optim.Adam(model_params, lr=CFG_LR_INIT)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        for iter in range(CFG_NUM_ITER):
            loss_all = {'kpts2d':0.0, 'depth':0.0, 'seg':0.0, 'reg':0.0, 'obj':0.0}
            if CFG_WITH_OBJ:
                obj_param = model_obj()

            num_skip = 0
            for camIdx, camID in enumerate(CFG_CAMID_SET):
                # skip non-detected camera
                if np.isnan(dataloader_set[camIdx][frame]['kpts3d']).any():
                    num_skip += 1
                    print("skip cam ", camID)
                    continue

                hand_param = model()

                if not CFG_WITH_OBJ:
                    losses = loss_func(pred=hand_param, pred_obj=None, render=flag_render,
                                       camIdx=camIdx, frame=frame)
                else:
                    ## TODO
                    losses = loss_func(pred=hand_param, pred_obj=obj_param, render=flag_render,
                                       camIdx=camIdx, frame=frame)

                for k in CFG_LOSS_DICT:
                    loss_all[k] += losses[k]
                # loss_func.visualize(CFG_SAVE_PATH, camID)

            num_done = len(CFG_CAMID_SET) - num_skip
            total_loss = sum(loss_all[k] for k in CFG_LOSS_DICT) / num_done
            logs = ["Iter: {}, Loss: {}".format(iter, total_loss.data)]
            logs += ['[%s:%.4f]' % (key, loss_all[key]/num_done) for key in loss_all.keys() if key in CFG_LOSS_DICT]
            logging.info(''.join(logs))

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

        ## visualization results of frame
        for camIdx, camID in enumerate(CFG_CAMID_SET):
            hand_param = model()

            if not CFG_WITH_OBJ:
                loss_func.visualize(pred=hand_param, pred_obj=None, camIdx=camIdx, frame=frame,
                                    save_path=CFG_SAVE_PATH, camID=camID, flag_obj=False, flag_crop=True)
            else:
                obj_param = model_obj()
                loss_func.visualize(pred=hand_param, pred_obj=obj_param, camIdx=camIdx, frame=frame,
                                    save_path=CFG_SAVE_PATH, camID=camID, flag_obj=True, flag_crop=True)

        print("end frame")
        cv2.waitKey(0)


if __name__ == "__main__":
    app.run(main)