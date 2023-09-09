import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(), 'modules'))
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
from utils.dataUtils import *


## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230905', 'target db name')   ## name ,default, help
flags.DEFINE_string('seq', '230905_S01_obj_30_grasp_01', 'target sequence name')
flags.DEFINE_integer('initNum', 1, 'initial frame num of trial_0, check mediapipe results')

FLAGS(sys.argv)



def main(argv):
    logging.get_absl_handler().setFormatter(None)

    save_num = 0

    flag_render = False
    if 'depth' in CFG_LOSS_DICT or 'seg' in CFG_LOSS_DICT:
        flag_render = True

    targetDir = os.path.join(CFG_DATA_DIR, FLAGS.db, FLAGS.seq)
    targetDir_result = os.path.join(CFG_DATA_DIR, FLAGS.db + '_result', FLAGS.seq)

    for trialIdx, trialName in enumerate(sorted(os.listdir(targetDir))):
        ## Load data of each camera, save pkl file for second run.
        print("loading data... %s %s " % (FLAGS.seq, trialName))
        mas_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'mas', CFG_DEVICE)
        sub1_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'sub1', CFG_DEVICE)
        sub2_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'sub2', CFG_DEVICE)
        sub3_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, 'sub3', CFG_DEVICE)

        ## Initialize renderer, every renderer's extrinsic is set to master camera extrinsic
        mas_K, mas_M, mas_D = mas_dataloader.cam_parameter
        default_M = np.eye(4)[:3]
        mas_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, default_M, mas_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
        sub1_K, _, sub1_D = sub1_dataloader.cam_parameter
        sub1_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, default_M, sub1_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
        sub2_K, _, sub2_D = sub2_dataloader.cam_parameter
        sub2_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, default_M, sub2_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
        sub3_K, _, sub3_D = sub3_dataloader.cam_parameter
        sub3_renderer = Renderer(CFG_DEVICE, CFG_BATCH_SIZE, default_M, sub3_K, (CFG_IMG_HEIGHT, CFG_IMG_WIDTH))

        dataloader_set = [mas_dataloader, sub1_dataloader, sub2_dataloader, sub3_dataloader]
        renderer_set = [mas_renderer, sub1_renderer, sub2_renderer, sub3_renderer]

        if (len(mas_dataloader) != len(sub1_dataloader)) or (len(mas_dataloader) != len(sub2_dataloader)) or (len(mas_dataloader) != len(sub3_dataloader)):
            raise ValueError("The number of data is not same between cameras")

        if CFG_WITH_OBJ:
            obj_dataloader = ObjectLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, mas_dataloader.cam_parameter)

        ## Initialize loss function
        loss_func = MultiViewLossFunc(device=CFG_DEVICE, dataloaders=dataloader_set, renderers=renderer_set, losses=CFG_LOSS_DICT)
        loss_func.set_main_cam(main_cam_idx=0)

        ## Initialize hand model
        model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE, side=CFG_MANO_SIDE)
        model.change_grads(root=True, rot=True, pose=True, shape=True)

        ## Initialize object model
        if CFG_WITH_OBJ:
            obj_template_mesh = obj_dataloader.obj_mesh_data
            model_obj = ObjModel(CFG_DEVICE, CFG_BATCH_SIZE, obj_template_mesh)
            loss_func.set_object_main_extrinsic(0)      #  Set object's main camera extrinsic as mas

        ## Start optimization per frame
        cfg_lr_init = CFG_LR_INIT
        for frame in range(len(mas_dataloader)):
            t1 = time.time()

            # check visualizeMP results in {YYMMDD} folder, define first frame on --initNum
            if trialIdx == 0 and frame < FLAGS.initNum:
                continue

            detected_cams = []
            for camIdx, camID in enumerate(CFG_CAMID_SET):
                if dataloader_set[camIdx][frame] is not None:
                    detected_cams.append(camIdx)
            if len(detected_cams) < 3:
                print('detected hand is less than 3, skip the frame ', frame)
                continue

            # set initial loss, early stopping threshold
            best_kps_loss = torch.inf
            prev_kps_loss = 0
            ealry_stopping_patience = 0
            ealry_stopping_patience_v2 = 0

            # set object init pose and marker pose as GT for projected vertex.
            if CFG_WITH_OBJ:
                obj_pose = obj_dataloader[frame][:-1, :]
                obj_pose[:3, -1] *= 0.1
                model_obj.update_pose(pose=obj_pose)

                marker_cam_pose = obj_dataloader.marker_cam_pose[str(frame)]     # marker 3d pose with camera coordinate(master)
                loss_func.set_object_marker_pose(marker_cam_pose, CFG_vertspermarker[obj_dataloader.obj_name])

            ## Initialize optimizer
            params_hand = set_lr_forHand(model, cfg_lr_init)
            if not CFG_WITH_OBJ:
                optimizer = torch.optim.Adam(params_hand)
            else:
                params_obj = set_lr_forObj(model_obj, CFG_LR_INIT_OBJ)
                params = list(params_hand) + list(params_obj)
                optimizer = torch.optim.Adam(params)

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-9)

            kps_loss = {}
            for iter in range(CFG_NUM_ITER):
                loss_all = {'kpts2d':0.0, 'depth':0.0, 'seg':0.0, 'reg':0.0, 'contact': 0.0}


                hand_param = model()
                if CFG_WITH_OBJ:
                    obj_param = model_obj()


                num_skip = 0
                for camIdx in detected_cams:

                    camID = CFG_CAMID_SET[camIdx]
                    # skip non-detected camera
                    if np.isnan(dataloader_set[camIdx].sample_kpt[frame]['kpts3d']).any():
                        num_skip += 1
                        print("skip cam ", camID)
                        continue

                    if not CFG_WITH_OBJ:
                        losses = loss_func(pred=hand_param, pred_obj=None, render=flag_render,
                                           camIdx=camIdx, frame=frame)
                    else:
                        losses = loss_func(pred=hand_param, pred_obj=obj_param, render=flag_render,
                                           camIdx=camIdx, frame=frame, contact=iter>(CFG_NUM_ITER-CFG_NUM_ITER_CONTACT))

                        # if camID == 'mas':
                        #     loss_func.visualize(pred=hand_param, pred_obj=obj_param, camIdx=camIdx, frame=frame,
                        #                     camID=camID, flag_obj=True, flag_crop=True)

                    for k in CFG_LOSS_DICT:
                        loss_all[k] += losses[k]


                num_done = len(CFG_CAMID_SET) - num_skip
                total_loss = sum(loss_all[k] for k in CFG_LOSS_DICT) / num_done

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()
                lr_scheduler.step()

                cur_kpt_loss = loss_all['kpts2d'].item() / num_done
                kps_loss[iter] = cur_kpt_loss
                logs = ["[{} - frame {}] Iter: {}, Loss: {:.4f}".format(trialName, frame, iter, total_loss.item())]
                logs += ['[%s:%.4f]' % (key, loss_all[key]/num_done) for key in loss_all.keys() if key in CFG_LOSS_DICT]
                logging.info(''.join(logs))



                ### sparse criterion on converge for v1 db release, need to be tight ###
                if abs(prev_kps_loss - cur_kpt_loss) < 0.5:
                    ealry_stopping_patience_v2 += 1

                if cur_kpt_loss < best_kps_loss:
                    best_kps_loss = cur_kpt_loss
                if cur_kpt_loss < CFG_LOSS_THRESHOLD:
                    ealry_stopping_patience += 1

                if cur_kpt_loss < CFG_LOSS_THRESHOLD and ealry_stopping_patience > CFG_PATIENCE:
                    logging.info('Early stopping(less than THRESHOLD) at iter %d' % iter)
                    break
                if ealry_stopping_patience_v2 > CFG_PATIENCE_v2:
                    logging.info('Early stopping(converged) at iter %d' % iter)
                    break

                prev_kps_loss = cur_kpt_loss

            hand_param = model()
            ### visualization results of frame
            for camIdx in detected_cams:
                camID = CFG_CAMID_SET[camIdx]

                save_path = os.path.join(targetDir_result, trialName, 'visualization', camID)
                os.makedirs(save_path, exist_ok=True)
                if not CFG_WITH_OBJ:
                    loss_func.visualize(pred=hand_param, pred_obj=None, camIdx=camIdx, frame=frame,
                                        save_path=save_path, camID=camID, flag_obj=False, flag_crop=True)
                else:
                    obj_param = model_obj()
                    loss_func.visualize(pred=hand_param, pred_obj=obj_param, camIdx=camIdx, frame=frame,
                                        save_path=save_path, camID=camID, flag_obj=True, flag_crop=True)
            cv2.waitKey(0)

            ### save annotation per frame as json format
            if CFG_WITH_OBJ:
                obj_param = [model_obj.get_object_mat(), obj_dataloader.obj_mesh_name]
            else:
                obj_param = [None, [None]]
            save_annotation(targetDir_result, trialName, frame,  FLAGS.seq, hand_param, obj_param, CFG_MANO_SIDE)
            t2 = time.time()
            print("end %s - frame %s, processed %s" % (trialName, frame, t2 - t1))

            save_num += 1
    print("end time : ", time.ctime())
    print("total processed frames : ", save_num)

if __name__ == "__main__":
    app.run(main)
