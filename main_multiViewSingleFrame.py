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
import multiprocessing
from cProfile import Profile
from pstats import Stats


## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230905', 'target db name')   ## name ,default, help
flags.DEFINE_string('seq', '230905_S01_obj_30_grasp_01', 'target sequence name')
flags.DEFINE_integer('initNum', 6, 'initial frame num of trial_0, check mediapipe results')

FLAGS(sys.argv)

def __update_global_pose__(model, model_obj, loss_func, detected_cams, frame, optimizer, trialName, iter=30):
    kps_loss = {}
    for iter in range(iter):
        t_iter = time.time()

        loss_all = {'kpts2d': 0.0, 'depth': 0.0, 'seg': 0.0, 'reg': 0.0, 'contact': 0.0, 'depth_rel': 0.0}
        loss_weight = {'kpts2d': 1.0, 'depth': 1.0, 'seg': 1.0, 'reg': 1.0, 'contact': 1.0, 'depth_rel': 1.0}

        hand_param = model()
        if CFG_WITH_OBJ:
            obj_param = model_obj()
        else:
            obj_param = None

        losses = loss_func(pred=hand_param, pred_obj=obj_param, camIdxSet=detected_cams, frame=frame)

        ### visualization for debug
        # loss_func.visualize(pred=hand_param, pred_obj=obj_param, frame=frame,
        #                     camIdxSet=[3], flag_obj=CFG_WITH_OBJ, flag_crop=True)

        for camIdx in detected_cams:
            for k in CFG_LOSS_DICT:
                loss_all[k] += losses[camIdx][k] * float(CFG_CAM_WEIGHT[camIdx])

        total_loss = sum(loss_all[k] * loss_weight[k] for k in CFG_LOSS_DICT) / len(detected_cams)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # lr_scheduler.step()

        cur_kpt_loss = loss_all['kpts2d'].item() / len(detected_cams)
        kps_loss[iter] = cur_kpt_loss

        iter_t = time.time() - t_iter
        logs = ["[{} - frame {}] [t : {:.2f} sec] Iter: {}, Loss: {:.4f}".format(trialName, frame, iter_t, iter,
                                                                                 total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                 key in CFG_LOSS_DICT]
        logging.info(''.join(logs))


def __update_all_pose__(model, model_obj, loss_func, detected_cams, frame, optimizer, trialName, iter=150):

    kps_loss = {}
    use_contact_loss = False

    # set initial loss, early stopping threshold
    best_kps_loss = torch.inf
    prev_kps_loss = 0
    ealry_stopping_patience = 0
    ealry_stopping_patience_v2 = 0

    for iter in range(iter):
        t_iter = time.time()

        loss_all = {'kpts2d': 0.0, 'depth': 0.0, 'seg': 0.0, 'reg': 0.0, 'contact': 0.0, 'depth_rel': 0.0}
        loss_weight = {'kpts2d': 1.0, 'depth': 1.0, 'seg': 1.0, 'reg': 1.0, 'contact': 1.0, 'depth_rel': 1.0}

        hand_param = model()
        if CFG_WITH_OBJ:
            obj_param = model_obj()
        else:
            obj_param = None

        losses = loss_func(pred=hand_param, pred_obj=obj_param, camIdxSet=detected_cams, frame=frame, contact=use_contact_loss)

        loss_func.visualize(pred=hand_param, pred_obj=obj_param, frame=frame, camIdxSet=[0, 3], flag_obj=CFG_WITH_OBJ, flag_crop=True)

        for camIdx in detected_cams:
            for k in CFG_LOSS_DICT:
                loss_all[k] += losses[camIdx][k] * float(CFG_CAM_WEIGHT[camIdx])

        total_loss = sum(loss_all[k] * loss_weight[k] for k in CFG_LOSS_DICT) / len(detected_cams)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # lr_scheduler.step()

        cur_kpt_loss = loss_all['kpts2d'].item() / len(detected_cams)
        kps_loss[iter] = cur_kpt_loss

        iter_t = time.time() - t_iter
        logs = ["[{} - frame {}] [t : {:.2f} sec] Iter: {}, Loss: {:.4f}".format(trialName, frame, iter_t, iter,
                                                                                 total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                 key in CFG_LOSS_DICT]
        logging.info(''.join(logs))

        ## criteria for contact loss
        if cur_kpt_loss < CFG_CONTACT_START_THRESHOLD:
            use_contact_loss = True

        ## sparse criterion on converge for v1 db release, need to be tight
        if CFG_EARLYSTOPPING:
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


def main(argv):
    logging.get_absl_handler().setFormatter(None)
    save_num = 0

    target_dir = os.path.join(CFG_DATA_DIR, FLAGS.db, FLAGS.seq)
    target_dir_result = os.path.join(CFG_DATA_DIR, FLAGS.db + '_result', FLAGS.seq)

    for trialIdx, trialName in enumerate(sorted(os.listdir(target_dir))):
        save_path = os.path.join(target_dir_result, trialName, "visualization")

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

        ## Initialize loss function
        loss_func = MultiViewLossFunc(device=CFG_DEVICE, dataloaders=dataloader_set, renderers=renderer_set, losses=CFG_LOSS_DICT).to(CFG_DEVICE)
        loss_func.set_main_cam(main_cam_idx=0)

        ## Initialize hand model
        model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE, side=CFG_MANO_SIDE).to(CFG_DEVICE)
        model_obj = None

        ## Initialize object dataloader & model
        if CFG_WITH_OBJ:
            obj_dataloader = ObjectLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.seq, trialName, mas_dataloader.cam_parameter)

            obj_template_mesh = obj_dataloader.obj_mesh_data
            model_obj = ObjModel(CFG_DEVICE, CFG_BATCH_SIZE, obj_template_mesh).to(CFG_DEVICE)
            loss_func.set_object_main_extrinsic(0)      #  Set object's main camera extrinsic as mas

        ## Start optimization per frame
        for frame in range(len(mas_dataloader)):
            t_start = time.time()

            # check visualizeMP results in {YYMMDD} folder, define first frame on --initNum
            if trialIdx == 0 and frame < FLAGS.initNum:
                continue

            ### skip the frame if detected hand is less than 3
            detected_cams = []
            for camIdx, camID in enumerate(CFG_CAMID_SET):
                if dataloader_set[camIdx][frame] is not None:
                    detected_cams.append(camIdx)
            if len(detected_cams) < 3:
                print('detected hand is less than 3, skip the frame ', frame)
                continue

            ### set object init pose and marker pose as GT for projected vertex.
            if CFG_WITH_OBJ:
                obj_pose = obj_dataloader[frame][:-1, :]
                obj_pose[:3, -1] *= 0.1
                model_obj.update_pose(pose=obj_pose)

                marker_cam_pose = obj_dataloader.marker_cam_pose[str(frame)]     # marker 3d pose with camera coordinate(master)
                loss_func.set_object_marker_pose(marker_cam_pose, CFG_vertspermarker[obj_dataloader.obj_name])

            ### initialize optimizer, scheduler
            optimizer = initialize_optimizer(model, model_obj, CFG_LR_INIT, CFG_WITH_OBJ, CFG_LR_INIT_OBJ)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-9)

            ### update global pos, rot first
            model.change_grads(root=True, rot=True, pose=False, shape=False, scale=False, tip=False)
            __update_global_pose__(model, model_obj, loss_func, detected_cams, frame, optimizer, trialName, iter=30)

            ### update all
            model.change_grads(root=True, rot=True, pose=True, shape=True, scale=True, tip=True)
            __update_all_pose__(model, model_obj, loss_func, detected_cams, frame, optimizer, trialName, iter=CFG_NUM_ITER)


            ### final result of frame
            pred_hand = model()
            if CFG_WITH_OBJ:
                pred_obj = model_obj()
                pred_obj_anno = [model_obj.get_object_mat().tolist(), obj_dataloader.obj_mesh_name]
            else:
                pred_obj = None
                pred_obj_anno = [None, None]

            ### visualization results of frame
            loss_func.visualize(pred=pred_hand, pred_obj=pred_obj, camIdxSet=detected_cams, frame=frame,
                                    save_path=save_path, flag_obj=CFG_WITH_OBJ, flag_crop=True)

            ### save annotation per frame as json format
            save_annotation(target_dir_result, trialName, frame,  FLAGS.seq, pred_hand, pred_obj_anno, CFG_MANO_SIDE)

            print("end %s - frame %s, processed %s" % (trialName, frame, time.time() - t_start))
            save_num += 1

            ####### debug #############
            cv2.waitKey(0)
            break

    print("end time : ", time.ctime())
    print("total processed frames : ", save_num)


if __name__ == "__main__":
    app.run(main)


