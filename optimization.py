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
from utils.modelUtils import initialize_optimizer, update_optimizer, initialize_optimizer_obj

from absl import flags
from absl import app
from absl import logging
from natsort import natsorted
from config import *
import time
from utils.dataUtils import *
import multiprocessing
from cProfile import Profile
from pstats import Stats
import pandas as pd


flag_debug_vis_glob = False
flag_debug_vis_part = False
flag_debug_vis_all = False

## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230915', 'target db name')   ## name ,default, help
flags.DEFINE_string('cam_db', '230915_cam', 'target db name')   ## name ,default, help
flags.DEFINE_integer('start', None, 'start idx of sequence(ordered)')
flags.DEFINE_integer('end', None, 'end idx of sequence(ordered)')

flags.DEFINE_integer('initNum', 0, 'initial frame num of trial_0, check mediapipe results')

# flags.DEFINE_string('seq', '230905_S02_obj_03_grasp_3', 'target sequence name')
## NO SPACE between sequences. --seq_list 230905_S02_obj_03_grasp_3,230905_S02_obj_03_grasp_3,..
# flags.DEFINE_string('seq_list', '230905_S02_obj_03_grasp_3', 'target sequence name')
flags.DEFINE_bool('headless', False, 'headless mode for visualization')
FLAGS(sys.argv)

# torch.autograd.set_detect_anomaly(True)


### check config date ###
CFG_DATE = None
if FLAGS.db in ['230829', '230830', '230904', '230905', '230906', '230907', '230908']:
    CFG_DATE = '230829~230908'
elif FLAGS.db in ['230909', '230910', '230911', '230912', '230913']:
    CFG_DATE = '230909~230913'
elif FLAGS.db in ['230914']:
    CFG_DATE = '230914'
elif FLAGS.db in ['230915', '230916', '230917', '230918', '230919', '230920', '230921',  '230922', '230923', '230924', '230925',  '230926', '230927']:
    CFG_DATE = '230915~'
else:
    assert False, 'no CFG date matches, contact KAIST'


def __update_global_wrist__(model, loss_func, detected_cams, frame, lr_init, target_seq, trialName, iter=40):

    loss_dict_global = ['kpts_palm', 'reg']

    model.change_grads_all(root=True, rot=False, pose=False, shape=False, scale=True)
    optimizer = initialize_optimizer(model, None, lr_init * 4.0, False, None)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    loss_weight = {'kpts_palm': 1.0, 'reg': 1.0}
    for iter in range(iter):
        t_iter = time.time()
        optimizer.zero_grad()

        loss_all = {'kpts_palm': 0.0, 'reg': 0.0}

        hand_param = model()
        obj_param = None

        losses, losses_single = loss_func(pred=hand_param, pred_obj=obj_param, camIdxSet=detected_cams, frame=frame,
                           loss_dict=loss_dict_global, flag_headless=FLAGS.headless)

        ### visualization for debug
        if flag_debug_vis_glob:
            loss_func.visualize(pred=hand_param, pred_obj=None, frame=frame, camIdxSet=[detected_cams[0]], flag_obj=False, flag_crop=True, flag_headless=FLAGS.headless)

        for camIdx in detected_cams:
            loss_cam = losses[camIdx]
            for key in loss_cam.keys():
                loss_all[key] += loss_cam[key] * float(CFG_CAM_WEIGHT[camIdx])

        for key in losses_single.keys():
            loss_all[key] += losses_single[key]

        total_loss = sum(loss_all[k] * loss_weight[k] for k in loss_dict_global) / len(detected_cams)

        total_loss.backward(retain_graph=True)
        optimizer.step()
        lr_scheduler.step()

        iter_t = time.time() - t_iter
        logs = ["[{} - {} - frame {}] [Global] Iter: {}, Loss: {:.4f}".format(target_seq, trialName, frame, iter, total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                 key in loss_dict_global]
        logging.info(''.join(logs))


def __update_global__(model, loss_func, detected_cams, frame, lr_init, target_seq, trialName, iter=40):

    loss_dict_global = ['kpts_palm', 'reg']

    model.change_grads_all(root=True, rot=True, pose=False, shape=False, scale=True)
    optimizer = initialize_optimizer(model, None, lr_init, False, None)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    loss_weight = {'kpts_palm': 1.0, 'reg': 1.0}
    for iter in range(iter):
        t_iter = time.time()
        optimizer.zero_grad()

        loss_all = {'kpts_palm': 0.0, 'reg': 0.0}

        hand_param = model()
        obj_param = None

        losses, losses_single = loss_func(pred=hand_param, pred_obj=obj_param, camIdxSet=detected_cams, frame=frame,
                           loss_dict=loss_dict_global, flag_headless=FLAGS.headless)

        ### visualization for debug
        if flag_debug_vis_glob:
            loss_func.visualize(pred=hand_param, pred_obj=None, frame=frame, camIdxSet=[detected_cams[0]], flag_obj=False, flag_crop=True, flag_headless=FLAGS.headless)

        for camIdx in detected_cams:
            loss_cam = losses[camIdx]
            for key in loss_cam.keys():
                loss_all[key] += loss_cam[key] * float(CFG_CAM_WEIGHT[camIdx])

        for key in losses_single.keys():
            loss_all[key] += losses_single[key]

        total_loss = sum(loss_all[k] * loss_weight[k] for k in loss_dict_global) / len(detected_cams)

        total_loss.backward(retain_graph=True)
        optimizer.step()
        lr_scheduler.step()

        iter_t = time.time() - t_iter
        logs = ["[{} - {} - frame {}] [Global] Iter: {}, Loss: {:.4f}".format(target_seq, trialName, frame, iter, total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                 key in loss_dict_global]
        logging.info(''.join(logs))


def __update_parts__(model, loss_func, detected_cams, frame, lr_init, target_seq, trialName, iterperpart=40):

    kps_loss = {}

    grad_order = [[True, False, False], [True, True, False], [True, True, True]]
    loss_dict_parts = ['kpts2d', 'reg']#, 'depth_rel']

    model.input_scale.data *= torch.FloatTensor([0.95]).to('cuda')
    for step in range(3):
        model.change_grads_parts(root=True, rot=True,
                                 pose_1=grad_order[step][0], pose_2=grad_order[step][1], pose_3=grad_order[step][2],
                                 shape=True, scale=True)

        optimizer = initialize_optimizer(model, None, lr_init, False, None)
        optimizer = update_optimizer(optimizer, ratio_root=0.5 ** step, ratio_rot=0.5 ** step, ratio_scale=0.5 ** step, ratio_pose=0.5 ** step)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

        loss_weight = {'kpts2d': 1.0, 'reg': 1.0, 'depth_rel': 1.0}
        for iter in range(iterperpart):
            t_iter = time.time()

            optimizer.zero_grad()

            loss_all = {'kpts2d': 0.0, 'reg': 0.0, 'depth_rel': 0.0}

            hand_param = model()
            obj_param = None

            losses, losses_single = loss_func(pred=hand_param, pred_obj=obj_param, camIdxSet=detected_cams, frame=frame, loss_dict=loss_dict_parts, parts=step, flag_headless=FLAGS.headless)
            if flag_debug_vis_part:
                loss_func.visualize(pred=hand_param, pred_obj=None, frame=frame, camIdxSet=detected_cams, flag_obj=False, flag_crop=True, flag_headless=FLAGS.headless)

            for camIdx in detected_cams:
                loss_cam = losses[camIdx]
                for key in loss_cam.keys():
                    loss_all[key] += loss_cam[key] * float(CFG_CAM_WEIGHT[camIdx])

            for key in losses_single.keys():
                loss_all[key] += losses_single[key]

            total_loss = sum(loss_all[k] * loss_weight[k] for k in loss_dict_parts) / len(detected_cams)

            total_loss.backward(retain_graph=True)
            optimizer.step()
            # lr_scheduler.step()

            cur_kpt_loss = loss_all['kpts2d'].item() / len(detected_cams)
            kps_loss[iter] = cur_kpt_loss

            iter_t = time.time() - t_iter
            logs = ["[{} - {} - frame {}] [Parts] Iter: {}, Loss: {:.4f}".format(target_seq, trialName, frame, iter,
                                                                                     total_loss.item())]
            logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                     key in loss_dict_parts]
            logging.info(''.join(logs))

def __update_all__(model, model_obj, loss_func, detected_cams, frame, lr_init, lr_init_obj, target_seq, trialName, iter=100):
    kps_loss = {}
    use_contact_loss = False
    use_penetration_loss = False

    # set initial loss, early stopping threshold
    best_loss = torch.inf
    prev_kps_loss = 0
    prev_obj_loss = torch.inf
    prev_depthseg_loss = 0
    early_stopping_patience = 0
    early_stopping_patience_v2 = 0
    early_stopping_patience_obj = 0

    # model.change_grads(root=True, rot=True, pose=True, shape=True, scale=False)

    model.change_grads_all(root=True, rot=True, pose=True, shape=True, scale=True)
    # optimizer = initialize_optimizer(model, model_obj, lr_init, CFG_WITH_OBJ, lr_init_obj)
    optimizer = initialize_optimizer(model, None, lr_init, False, None)
    optimizer = update_optimizer(optimizer, ratio_root=0.1, ratio_rot=0.1, ratio_shape=1.0, ratio_scale=0.1, ratio_pose=0.2)

    if CFG_WITH_OBJ:
        optimizer_obj = initialize_optimizer_obj(model_obj, lr_init_obj)
        flag_update_obj = True
        lr_scheduler_obj = torch.optim.lr_scheduler.StepLR(optimizer_obj, step_size=5, gamma=0.95)

    loss_weight = CFG_LOSS_WEIGHT
    loss_weight['kpts2d'] = 0.8
    model.input_scale.data *= torch.FloatTensor([0.95]).to('cuda')

    for iter in range(iter):
        t_iter = time.time()

        optimizer.zero_grad()
        if CFG_WITH_OBJ:
            optimizer_obj.zero_grad()
        loss_all = {'kpts2d': 0.0, 'depth': 0.0, 'seg': 0.0, 'reg': 0.0, 'contact': 0.0, 'penetration': 0.0,
                    'depth_rel': 0.0, 'temporal': 0.0, 'kpts_tip': 0.0, 'depth_obj': 0.0, 'seg_obj': 0.0, 'pose_obj':0.0}

        hand_param = model()
        if CFG_WITH_OBJ:
            obj_param = model_obj()
        else:
            obj_param = None
            
        losses, losses_single, contact = loss_func(pred=hand_param, pred_obj=obj_param, camIdxSet=detected_cams,
                                                   frame=frame, loss_dict=CFG_LOSS_DICT, contact=use_contact_loss,
                                                   penetration=use_penetration_loss, flag_headless=FLAGS.headless)

        model.contact = contact
        if flag_debug_vis_all:
            loss_func.visualize(pred=hand_param, pred_obj=obj_param, frame=frame, camIdxSet=detected_cams, flag_obj=CFG_WITH_OBJ,
                                flag_crop=True, flag_headless=FLAGS.headless)

        ## apply cam weight
        for camIdx in detected_cams:
            loss_cam = losses[camIdx]
            for key in loss_cam.keys():
                loss_all[key] += loss_cam[key] * float(CFG_CAM_WEIGHT[camIdx])

        for key in losses_single.keys():
            loss_all[key] += losses_single[key]

        ## apply loss weight
        total_loss = sum(loss_all[k] * loss_weight[k] for k in CFG_LOSS_DICT) / len(detected_cams)
        total_loss.backward(retain_graph=True)

        optimizer.step()
        if CFG_WITH_OBJ and flag_update_obj:
            optimizer_obj.step()
            lr_scheduler_obj.step()
        # lr_scheduler.step()


        # iter_t = time.time() - t_iter
        # print("iter_t : ", iter_t)
        logs = ["[{} - {} - frame {}] [All] Iter: {}, Loss: {:.4f}".format(target_seq, trialName, frame, iter, total_loss.item())]
        logs += ['[%s:%.4f]' % (key, loss_all[key] / len(detected_cams)) for key in loss_all.keys() if
                 key in CFG_LOSS_DICT]
        logging.info(''.join(logs))


        cur_kpt_loss = loss_all['kpts2d'].item() / len(detected_cams)
        kps_loss[iter] = cur_kpt_loss
        # cur_depthseg_loss = (loss_all['depth'].item() + loss_all['seg'].item()) / len(detected_cams)

        if CFG_WITH_OBJ:
            cur_obj_loss = loss_all['depth_obj'].item() / len(detected_cams)

            if prev_obj_loss < cur_obj_loss:
                early_stopping_patience_obj += 1
            if prev_obj_loss > cur_obj_loss:
                early_stopping_patience_obj = 0

            if early_stopping_patience_obj > CFG_PATIENCE_obj:
                # flag_update_obj = False
                ## criteria for contact loss
                if 'contact' in CFG_LOSS_DICT:
                    use_contact_loss = True
                if 'penetration' in CFG_LOSS_DICT:
                    use_penetration_loss = True


        ## sparse criterion on converge for v1 db release, need to be tight
        if CFG_EARLYSTOPPING:
            if abs(prev_kps_loss - cur_kpt_loss) < 2.0:# or abs(prev_depthseg_loss - cur_depthseg_loss) < 10.0:
                early_stopping_patience_v2 += 1

            if cur_kpt_loss < best_loss:
                best_loss = cur_kpt_loss
            if cur_kpt_loss < CFG_LOSS_THRESHOLD:
                early_stopping_patience += 1

            if cur_kpt_loss < CFG_LOSS_THRESHOLD and early_stopping_patience > CFG_PATIENCE:
                logging.info('Early stopping(less than THRESHOLD) at iter %d' % iter)
                break
            if early_stopping_patience_v2 > CFG_PATIENCE_v2:
                logging.info('Early stopping(converged) at iter %d' % iter)
                break

            prev_kps_loss = cur_kpt_loss

        if CFG_WITH_OBJ:
            prev_obj_loss = cur_obj_loss
            # prev_depthseg_loss = cur_depthseg_loss


def main(argv):
    baseDir = os.path.join(os.getcwd(), 'dataset')

    assert FLAGS.db == FLAGS.cam_db[:6], "wrong db-cam_db pair. check name"
    assert os.path.exists(os.path.join(baseDir, FLAGS.db)), "no {YYMMDD} directory. check."
    assert os.path.exists(os.path.join(baseDir, FLAGS.cam_db)), "no{YYMMDD}_cam directory. check."
    assert os.path.exists(os.path.join(baseDir, FLAGS.db + '_obj')), "no {YYMMDD}_obj directory. check."
    assert os.path.exists(
        os.path.join(baseDir, 'obj_scanned_models')), "no dataset/obj_scanned_models directory. check."
    assert os.path.exists(
        os.path.join(os.getcwd(), 'modules/deepLabV3plus/checkpoints')), "no segmentation checkpoint folder. check."

    t0 = time.time()
    logging.get_absl_handler().setFormatter(None)
    save_num = 0

    seq_list = natsorted(os.listdir(os.path.join(CFG_DATA_DIR, FLAGS.db)))
    if FLAGS.start != None and FLAGS.end != None:
        seq_list = seq_list[FLAGS.start:FLAGS.end]

    obj_unvalid_trials = []
    for target_seq in seq_list:
        target_dir = os.path.join(CFG_DATA_DIR, FLAGS.db, target_seq)
        target_dir_result = os.path.join(CFG_DATA_DIR, FLAGS.db + '_result', target_seq)

        for trialIdx, trialName in enumerate(sorted(os.listdir(target_dir))):
            save_path = os.path.join(target_dir_result, trialName, "visualization")
            log_path = os.path.join(target_dir_result, trialName, 'log')
            if not os.path.isdir(log_path):
                os.makedirs(log_path)

            ## Load data of each camera, save pkl file for second run.
            print("loading data... %s %s " % (target_seq, trialName))
            mas_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.cam_db, target_seq, trialName, 'mas', CFG_DEVICE)
            sub1_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.cam_db, target_seq, trialName, 'sub1', CFG_DEVICE)
            sub2_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.cam_db, target_seq, trialName, 'sub2', CFG_DEVICE)
            sub3_dataloader = DataLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.cam_db, target_seq, trialName, 'sub3', CFG_DEVICE)

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

            # if (len(mas_dataloader) != len(sub1_dataloader)) or (len(mas_dataloader) != len(sub2_dataloader)) or (len(mas_dataloader) != len(sub3_dataloader)):
            #     raise ValueError("The number of data is not same between cameras")

            ## Initialize loss function
            loss_func = MultiViewLossFunc(device=CFG_DEVICE, dataloaders=dataloader_set, renderers=renderer_set, losses=CFG_LOSS_DICT).to(CFG_DEVICE)
            loss_func.set_main_cam(main_cam_idx=0)

            ## Initialize hand model
            model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE, side=CFG_MANO_SIDE).to(CFG_DEVICE)
            model_obj = None

            ## Initialize object dataloader & model
            if CFG_WITH_OBJ:
                obj_dataloader = ObjectLoader(CFG_DATA_DIR, FLAGS.db, target_seq, trialName, mas_dataloader.cam_parameter)
                if obj_dataloader.quit:
                    print("unvalid obj pose, skip trial")
                    obj_unvalid_trials.append(target_seq + '_' + trialName)
                    continue

                obj_template_mesh = obj_dataloader.obj_mesh_data
                model_obj = ObjModel(CFG_DEVICE, CFG_BATCH_SIZE, obj_template_mesh).to(CFG_DEVICE)
                loss_func.set_object_main_extrinsic(0)      #  Set object's main camera extrinsic as mas

            flag_start = True
            flag_skip = 0

            loss_func.reset_prev_pose()
            loss_func.set_for_evaluation()

            eval_num = 0
            ## Start optimization per frame
            for frame in range(len(mas_dataloader)):
                t_start = time.time()

                ## check visualizeMP results in {YYMMDD} folder, use for debugging
                if trialIdx == 0 and frame < FLAGS.initNum:
                    continue

                ## if prev frame has tip GT, increase current frame's temporal loss
                if frame > 0 and frame % CFG_tipGT_interval == 1:
                    print("increase temp weight")
                    loss_func.temp_weight = CFG_temporal_loss_weight * 10.0
                else:
                    loss_func.temp_weight = CFG_temporal_loss_weight

                ## skip the frame if detected hand is less than 3
                detected_cams = []
                valid_cam_list = CFG_VALID_CAM

                # if trialName == '230905_S01_obj_16_grasp_14' and trialIdx == 0:
                #     valid_cam_list = ['mas', 'sub2', 'sub3']
                # if trialName == '230905_S01_obj_16_grasp_27' and trialIdx == 0:
                #     valid_cam_list = ['mas', 'sub2', 'sub3']

                for camIdx, camID in enumerate(valid_cam_list):
                    if 'bb' in dataloader_set[camIdx][frame].keys():
                        detected_cams.append(camIdx)
                if len(detected_cams) < 2:
                    print('detected hand is less than 2, skip the frame ', frame)
                    flag_skip += 1
                    continue

                ## reset previous pose data if skipped multiple frames
                if flag_skip > 3:
                    flag_skip = 0
                    loss_func.reset_prev_pose()
                    flag_start = True

                ## set object init pose and marker pose as GT for projected vertex.
                if CFG_WITH_OBJ:
                    if frame > len(obj_dataloader):
                        print('no obj pose')
                        continue
                    obj_pose = obj_dataloader[frame]
                    if obj_pose is None:
                        print('no obj pose')
                        continue
                    obj_pose = obj_pose[:-1, :]
                    # obj_pose[:3, -1] *= 0.1
                    model_obj.update_pose(pose=obj_pose)

                    marker_cam_pose = obj_dataloader.marker_cam_pose[str(frame)]     # marker 3d pose with camera coordinate(master)
                    marker_valid_idx = obj_dataloader.marker_valid_idx[str(frame)]
                    loss_func.set_object_marker_pose(marker_cam_pose, marker_valid_idx, obj_dataloader.obj_class, CFG_DATE, obj_dataloader.grasp_idx)

                ### initialize optimizer, scheduler
                lr_init = CFG_LR_INIT * 0.2
                lr_init_obj = CFG_LR_INIT_OBJ

                if flag_start:
                    lr_init *= 5.0
                    flag_start = False

                ### update global pose
                """
                    loss : 'kpts_palm' ~ multi-view 2D kpts loss for palm joints (0, 2, 3, 4)
                    target : wrist pose/rot, hand scale
                    except : hand shape, hand pose 
                """
                __update_global_wrist__(model, loss_func, detected_cams, frame,
                                  lr_init, target_seq, trialName)
                __update_global__(model, loss_func, detected_cams, frame,
                                  lr_init, target_seq, trialName)

                ### update incrementally
                """
                    loss : 'kpts2d', 'reg', 'depth_rel'
                        ~ multi-view 2D kpts loss for each set of hand parts(wrist to tip)                       
                    target : wrist pose/rot, hand scale, hand pose(each part) 
                    except : hand shape
                """
                __update_parts__(model, loss_func, detected_cams, frame,
                                 lr_init, target_seq, trialName, iterperpart=40)

                ### update all
                __update_all__(model, model_obj, loss_func, detected_cams, frame,
                               lr_init, lr_init_obj, target_seq, trialName, iter=CFG_NUM_ITER)

                # update prev pose if temporal loss activated
                pred_hand = model()
                if 'temporal' in CFG_LOSS_DICT:
                    loss_func.prev_hand_pose = pred_hand['pose'].clone().detach()
                    loss_func.prev_hand_shape = pred_hand['shape'].clone().detach()

                ### final result of frame
                if CFG_WITH_OBJ:
                    pred_obj = model_obj()
                    pred_obj_anno = [model_obj.get_object_mat().tolist(), obj_dataloader.obj_mesh_name]
                else:
                    pred_obj = None
                    pred_obj_anno = [None, None]

                ### visualization results of frame
                loss_func.visualize(pred=pred_hand, pred_obj=pred_obj, camIdxSet=detected_cams, frame=frame,
                                        save_path=save_path, flag_obj=CFG_WITH_OBJ, flag_crop=True, flag_headless=FLAGS.headless)

                loss_func.evaluation(pred_hand, pred_obj, detected_cams, frame)

                ### save annotation per frame as json format
                save_annotation(target_dir_result, trialName, frame,  target_seq, pred_hand, pred_obj_anno, CFG_MANO_SIDE)

                print("end %s - frame %s, processed %s" % (trialName, frame, time.time() - t_start))
                save_num += 1
                eval_num += 1

            loss_func.save_evaluation(log_path, eval_num)

            if eval_num != 0:
                # extract top 'num' indexes from depth f1 score and save as json
                top_index = loss_func.filtering_top_quality_index(num=60).tolist()
                p = os.path.join(target_dir_result, trialName)
                with open(os.path.join(p, 'top_60_index.json'), 'w') as f:
                    json.dump(top_index, f, ensure_ascii=False)

            del mas_dataloader
            del sub1_dataloader
            del sub2_dataloader
            del sub3_dataloader

            del mas_renderer
            del sub1_renderer
            del sub2_renderer
            del sub3_renderer

            del model
            del model_obj
            del loss_func

            torch.cuda.empty_cache()


    print("total processed time(min) : ", round((time.time() - t0) / 60., 2))
    print("total processed frames : ", save_num)

    print("(fill in google sheets) unvalid trials with wrong object pose data : ", obj_unvalid_trials)

if __name__ == "__main__":
    app.run(main)


