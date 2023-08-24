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
from utils.modelUtils import initialize_optimizer, set_lr_forHand, set_lr_forObj

from absl import flags
from absl import app
from absl import logging

from config import *
import time
import json


## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230822', 'target db name')   ## name ,default, help
flags.DEFINE_string('seq', '230822_S01_obj_01_grasp_13', 'target sequence name')
flags.DEFINE_string('objClass', 'banana', 'target object name')
FLAGS(sys.argv)

def save_annotation(targetDir, trialName, frame):
    anno_path = os.join(targetDir, trialName, 'annotation', 'anno_' + str(frame) + '.json')
    ### load current annotation(include updated meta info.)
    with open(anno_path, 'r') as file:
        anno = json.load(anno_path)

    ### update annotation
    # anno[annotations], anno[Mesh]
    anno['annotations'][0]['id'] = None
    anno['Mesh'][0]['id'] = None


    ### save full annotation
    with open(anno_path, 'w', encoding='utf-8') as file:
        json.dump(anno, file, indent='\t')



def main(argv):
    flag_render = False
    if 'depth' in CFG_LOSS_DICT or 'seg' in CFG_LOSS_DICT:
        flag_render = True

    targetDir = os.path.join(CFG_DATA_DIR, FLAGS.db, FLAGS.seq)
    for trialIdx, trialName in enumerate(sorted(os.listdir(targetDir))):
        ## Load data of each camera, save pkl file for second run.
        print("loading data... %s %s " % (FLAGS.seq, trialName))
        mas_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'mas')
        sub1_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'sub1')
        sub2_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'sub2')
        sub3_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'sub3')

        if CFG_WITH_OBJ:
            obj_dataloader = ObjectLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, FLAGS.objClass)

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

        if CFG_WITH_OBJ:
            obj_init_pose = obj_dataloader[0]   # numpy (4, 4) format
            obj_template_mesh = obj_dataloader.obj_mesh_data
            model_obj = ObjModel(CFG_DEVICE, CFG_BATCH_SIZE, obj_template_mesh, obj_init_pose)
            ## Set object's main camera extrinsic
            obj_main_cam_idx = CFG_CAMID_SET.index(obj_dataloader.obj_view)
            loss_func.set_object_main_extrinsic(obj_main_cam_idx)

        cfg_lr_init = CFG_LR_INIT

        for frame in range(len(mas_dataloader)):
            best_kps_loss = torch.inf
            ealry_stopping_patience = 0

            ## initial frames are often errorneous, check

            ## Currently, set object pose as init pose on every frame
            if CFG_WITH_OBJ:
                obj_pose = obj_dataloader[frame]
                model_obj.update_pose(pose=obj_pose)

            ## Initialize optimizer
            # TODO : skipped shape lr in model_params. check lr ratio with above definition
            if not CFG_WITH_OBJ:
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg_lr_init)
            else:
                params_hand = set_lr_forHand(model, cfg_lr_init)
                params_obj = set_lr_forObj(model_obj, CFG_LR_INIT_OBJ)

                params = list(params_hand) + list(params_obj)
                optimizer = torch.optim.Adam(params)

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-9)
            kps_loss = {}
            for iter in range(CFG_NUM_ITER):
                loss_all = {'kpts2d':0.0, 'depth':0.0, 'seg':0.0, 'reg':0.0, 'depth_obj':0.0}
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

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()
                lr_scheduler.step()
                cur_kpt_loss = loss_all['kpts2d'].item() / num_done
                kps_loss[iter] = cur_kpt_loss
                logs = ["Iter: {}, Loss: {}".format(iter, total_loss.item())]
                logs += ['[%s:%.4f]' % (key, loss_all[key]/num_done) for key in loss_all.keys() if key in CFG_LOSS_DICT]
                logging.info(''.join(logs))

                if cur_kpt_loss < best_kps_loss:
                    best_kps_loss = cur_kpt_loss
                else:
                    if cur_kpt_loss < CFG_LOSS_THRESHOLD:
                        ealry_stopping_patience += 1

                if ealry_stopping_patience > CFG_PATIENCE and cur_kpt_loss < CFG_LOSS_THRESHOLD:
                    logging.info('Early stopping at iter %d' % iter)
                    break


            ### temp draw loss graph
            # plt.plot(list(kps_loss.keys()), list(kps_loss.values()))
            # plt.savefig(CFG_SAVE_PATH + "/loss" + "/%d_loss_graph.png"%frame)
            # plt.close()
            # cfg_lr_init = 1e-2


            ### visualization results of frame
            for camIdx, camID in enumerate(CFG_CAMID_SET):
                hand_param = model()

                if not CFG_WITH_OBJ:
                    loss_func.visualize(pred=hand_param, pred_obj=None, camIdx=camIdx, frame=frame,
                                        save_path=CFG_SAVE_PATH, camID=camID, flag_obj=False, flag_crop=True)
                else:
                    obj_param = model_obj()
                    loss_func.visualize(pred=hand_param, pred_obj=obj_param, camIdx=camIdx, frame=frame,
                                        save_path=CFG_SAVE_PATH, camID=camID, flag_obj=True, flag_crop=True)

            ### save annotation per frame as json format
            save_annotation(targetDir, trialName, frame)

            print("end frame")
            cv2.waitKey(0)


if __name__ == "__main__":
    app.run(main)