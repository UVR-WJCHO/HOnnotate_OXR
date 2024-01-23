import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
sys.path.insert(0,os.path.join(os.getcwd(), 'modules'))
import torch
import torch.nn as nn
import numpy as np
import cv2

from modules.dataloader import DataLoader, ObjectLoader, DBLoader
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

from manopth.manolayer import ManoLayer
from utils.lossUtils import *


## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'NIA_db', 'target db name')   ## name ,default, help
flags.DEFINE_integer('start', None, 'start idx of sequence(ordered)')
flags.DEFINE_integer('end', None, 'end idx of sequence(ordered)')
flags.DEFINE_string('obj_db', 'obj_scanned_models_230915~', 'target obj_scanned_models folder')   ## obj_scanned_models_~230908
flags.DEFINE_bool('headless', False, 'headless mode for visualization')
FLAGS(sys.argv)


def mano3DToCam3D(xyz3D, ext):
    device = xyz3D.device
    xyz3D = torch.squeeze(xyz3D)
    ones = torch.ones((xyz3D.shape[0], 1), device=device)

    xyz4Dcam = torch.cat([xyz3D, ones], axis=1)
    # world to target cam
    xyz3Dcam2 = xyz4Dcam @ ext.T  # [:, :3]

    return xyz3Dcam2


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    uv = torch.matmul(K, xyz.transpose(2, 1)).transpose(2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]



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


def main(argv):
    baseDir = os.path.join(os.getcwd(), 'dataset')

    assert os.path.exists(os.path.join(baseDir, FLAGS.db)), "no {YYMMDD} directory. check."
    assert os.path.exists(
        os.path.join(baseDir, FLAGS.obj_db)), "no dataset/obj_scanned_models directory. check."
    assert os.path.exists(
        os.path.join(os.getcwd(), 'modules/deepLabV3plus/checkpoints')), "no segmentation checkpoint folder. check."

    mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
    mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                           center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(CFG_DEVICE)

    t0 = time.time()
    logging.get_absl_handler().setFormatter(None)
    save_num = 0

    seq_list = natsorted(os.listdir(os.path.join(CFG_DATA_DIR, FLAGS.db, 'source')))
    if FLAGS.start != None and FLAGS.end != None:
        seq_list = seq_list[FLAGS.start:FLAGS.end]

    for target_seq in seq_list:
        dir_source = os.path.join(CFG_DATA_DIR, FLAGS.db, 'source', target_seq)
        dir_annotation = os.path.join(CFG_DATA_DIR, FLAGS.db, 'annotation', target_seq)

        for trialIdx, trialName in enumerate(sorted(os.listdir(dir_source))):
            path_rgb = os.path.join(dir_source, trialName, 'rgb')
            path_depth = os.path.join(dir_source, trialName, 'depth')
            path_anno = os.path.join(dir_annotation, trialName, 'annotation')
            path_vis = os.path.join(dir_annotation, trialName, 'visualization')

            ## Load data of each camera, save pkl file for second run.
            print("loading data... %s %s " % (target_seq, trialName))
            dbloader = DBLoader(CFG_DATA_DIR, FLAGS.db, FLAGS.obj_db, target_seq, trialName, mano_layer, CFG_DEVICE)


            for camID in CFG_CAMID_SET:
                for idx in range(dbloader.db_len_dict[camID]):
                    data = dbloader[camID, idx]
                    renderer = dbloader.renderer_dict[camID]
                    Ms = dbloader.Ms_dict[camID]
                    Ks = dbloader.Ks_dict[camID]

                    hand_joints = data['mano_joints']
                    hand_verts = data['mano_verts']
                    obj_verts_world = data['obj_verts']

                    rgb_input = data['rgb']
                    depth_input = data['depth']

                    joints_cam = torch.unsqueeze(torch.Tensor(mano3DToCam3D(hand_joints, Ms)), axis=0)
                    verts_cam = torch.unsqueeze(mano3DToCam3D(hand_verts, Ms), 0)
                    verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, Ms), 0)

                    ## mesh rendering
                    pred_rendered = renderer.render_meshes([verts_cam, verts_cam_obj],
                                                                       [dbloader.hand_faces_template, dbloader.obj_faces_template],
                                                                       flag_rgb=True)
                    # pred_rendered_hand_only = renderer.render(verts_cam, dbloader.hand_faces_template, flag_rgb=True)
                    # pred_rendered_obj_only = renderer.render(verts_cam_obj, dbloader.obj_faces_template,
                    #                                                      flag_rgb=True)

                    rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                    depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
                    seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy())
                    seg_mesh = np.array(np.ceil(seg_mesh / np.max(seg_mesh)), dtype=np.uint8)

                    ## projection on image plane
                    pred_kpts2d = projectPoints(joints_cam, Ks)
                    pred_kpts2d = np.squeeze(pred_kpts2d.clone().cpu().detach().numpy())
                    rgb_2d_pred = paint_kpts(None, rgb_mesh, pred_kpts2d)


                    seg_mask = np.copy(seg_mesh)
                    seg_mask[seg_mesh > 0] = 1
                    rgb_2d_pred *= seg_mask[..., None]

                    ### reproduced visualization result ###
                    img_blend_pred = cv2.addWeighted(rgb_input, 1.0, rgb_2d_pred, 0.4, 0)
                    cv2.imshow("visualize", img_blend_pred)
                    cv2.waitKey(0)

    #
    #
    #
    #         # if (len(mas_dataloader) != len(sub1_dataloader)) or (len(mas_dataloader) != len(sub2_dataloader)) or (len(mas_dataloader) != len(sub3_dataloader)):
    #         #     raise ValueError("The number of data is not same between cameras")
    #
    #         ## Initialize loss function
    #         loss_func = MultiViewLossFunc(device=CFG_DEVICE, dataloaders=dataloader_set, renderers=renderer_set, losses=CFG_LOSS_DICT).to(CFG_DEVICE)
    #         loss_func.set_main_cam(main_cam_idx=0)
    #
    #         ## Initialize hand model
    #         model = HandModel(CFG_MANO_PATH, CFG_DEVICE, CFG_BATCH_SIZE, side=CFG_MANO_SIDE).to(CFG_DEVICE)
    #         model_obj = None
    #
    #         ## Initialize object dataloader & model
    #         if CFG_WITH_OBJ:
    #             obj_dataloader = ObjectLoader(CFG_DATA_DIR, FLAGS.db, target_seq, trialName, mas_dataloader.cam_parameter, FLAGS.obj_db)
    #             if obj_dataloader.quit:
    #                 print("unvalid obj pose, skip trial")
    #                 obj_unvalid_trials.append(target_seq + '_' + trialName)
    #                 continue
    #
    #             obj_template_mesh = obj_dataloader.obj_mesh_data
    #             model_obj = ObjModel(CFG_DEVICE, CFG_BATCH_SIZE, obj_template_mesh).to(CFG_DEVICE)
    #             loss_func.set_object_main_extrinsic(0)      #  Set object's main camera extrinsic as mas
    #
    #         flag_start = True
    #         flag_skip = 0
    #
    #         loss_func.reset_prev_pose()
    #         loss_func.set_for_evaluation()
    #
    #         eval_num = 0
    #         ## Start optimization per frame
    #         for frame in range(len(mas_dataloader)):
    #             torch.cuda.empty_cache()
    #             t_start = time.time()
    #
    #             ## check visualizeMP results in {YYMMDD} folder, use for debugging
    #             if trialIdx == 0 and frame < FLAGS.initNum:
    #                 continue
    #
    #             ## if prev frame has tip GT, increase current frame's temporal loss
    #             if frame > 0 and frame % CFG_tipGT_interval == 1:
    #                 print("increase temp weight")
    #                 loss_func.temp_weight = CFG_temporal_loss_weight * 10.0
    #             else:
    #                 loss_func.temp_weight = CFG_temporal_loss_weight
    #
    #             ## skip the frame if detected hand is less than 3
    #             detected_cams = []
    #             valid_cam_list = CFG_VALID_CAM
    #
    #             # if trialName == '230905_S01_obj_16_grasp_14' and trialIdx == 0:
    #             #     valid_cam_list = ['mas', 'sub2', 'sub3']
    #             # if trialName == '230905_S01_obj_16_grasp_27' and trialIdx == 0:
    #             #     valid_cam_list = ['mas', 'sub2', 'sub3']
    #
    #             for camIdx, camID in enumerate(valid_cam_list):
    #                 if dataloader_set[camIdx][frame] is None:
    #                     continue
    #                 if 'bb' in dataloader_set[camIdx][frame].keys():
    #                     detected_cams.append(camIdx)
    #             if len(detected_cams) < 2:
    #                 print('detected hand is less than 2, skip the frame ', frame)
    #                 flag_skip += 1
    #                 continue
    #
    #             ## reset previous pose data if skipped multiple frames
    #             if flag_skip > 3:
    #                 flag_skip = 0
    #                 loss_func.reset_prev_pose()
    #                 flag_start = True
    #
    #             ## set object init pose and marker pose as GT for projected vertex.
    #             if CFG_WITH_OBJ:
    #                 if frame > len(obj_dataloader):
    #                     print('no obj pose')
    #                     continue
    #                 obj_pose = obj_dataloader[frame]
    #                 if obj_pose is None or len(obj_pose.shape) != 2:
    #                     print('no obj pose')
    #                     continue
    #                 obj_pose = obj_pose[:-1, :]
    #                 # obj_pose[:3, -1] *= 0.1
    #                 model_obj.update_pose(pose=obj_pose)
    #
    #                 marker_cam_pose = obj_dataloader.marker_cam_pose[str(frame)]     # marker 3d pose with camera coordinate(master)
    #                 marker_valid_idx = obj_dataloader.marker_valid_idx[str(frame)]
    #                 loss_func.set_object_marker_pose(marker_cam_pose, marker_valid_idx, obj_dataloader.obj_class, CFG_DATE, obj_dataloader.grasp_idx)
    #
    #             ### initialize optimizer, scheduler
    #             lr_init = CFG_LR_INIT * 0.2
    #             lr_init_obj = CFG_LR_INIT_OBJ
    #
    #             if flag_start:
    #                 lr_init *= 5.0
    #                 flag_start = False
    #
    #             ### update global pose
    #             """
    #                 loss : 'kpts_palm' ~ multi-view 2D kpts loss for palm joints (0, 2, 3, 4)
    #                 target : wrist pose/rot, hand scale
    #                 except : hand shape, hand pose
    #             """
    #             __update_global_wrist__(model, loss_func, detected_cams, frame,
    #                               lr_init, target_seq, trialName)
    #             __update_global__(model, loss_func, detected_cams, frame,
    #                               lr_init, target_seq, trialName)
    #
    #             ### update incrementally
    #             """
    #                 loss : 'kpts2d', 'reg', 'depth_rel'
    #                     ~ multi-view 2D kpts loss for each set of hand parts(wrist to tip)
    #                 target : wrist pose/rot, hand scale, hand pose(each part)
    #                 except : hand shape
    #             """
    #             __update_parts__(model, loss_func, detected_cams, frame,
    #                              lr_init, target_seq, trialName, iterperpart=40)
    #
    #             ### update all
    #             __update_all__(model, model_obj, loss_func, detected_cams, frame,
    #                            lr_init, lr_init_obj, target_seq, trialName, iter=CFG_NUM_ITER)
    #
    #             # update prev pose if temporal loss activated
    #             pred_hand = model()
    #             if 'temporal' in CFG_LOSS_DICT:
    #                 loss_func.prev_hand_pose = pred_hand['pose'].clone().detach()
    #                 loss_func.prev_hand_shape = pred_hand['shape'].clone().detach()
    #
    #             ### final result of frame
    #             if CFG_WITH_OBJ:
    #                 pred_obj = model_obj()
    #                 pred_obj_anno = [model_obj.get_object_mat().tolist(), obj_dataloader.obj_mesh_name]
    #             else:
    #                 pred_obj = None
    #                 pred_obj_anno = [None, None]
    #
    #             ### visualization results of frame
    #             loss_func.visualize(pred=pred_hand, pred_obj=pred_obj, camIdxSet=detected_cams, frame=frame,
    #                                     save_path=save_path, flag_obj=CFG_WITH_OBJ, flag_crop=True, flag_headless=FLAGS.headless)
    #             loss_func.evaluation(pred_hand, pred_obj, detected_cams, frame)
    #
    #             ### save annotation per frame as json format
    #             save_annotation(target_dir_result, trialName, frame,  target_seq, pred_hand, pred_obj_anno, CFG_MANO_SIDE)
    #
    #             print("end %s - frame %s, processed %s" % (trialName, frame, time.time() - t_start))
    #             save_num += 1
    #             eval_num += 1
    #
    #         loss_func.save_evaluation(log_path, eval_num)
    #
    #         if eval_num != 0:
    #             # extract top 'num' indexes from depth f1 score and save as json
    #             top_index = loss_func.filtering_top_quality_index(num=60).tolist()
    #             p = os.path.join(target_dir_result, trialName)
    #             with open(os.path.join(p, 'top_60_index.json'), 'w') as f:
    #                 json.dump(top_index, f, ensure_ascii=False)
    #
    #         del mas_dataloader.sample_dict
    #         del sub1_dataloader.sample_dict
    #         del sub2_dataloader.sample_dict
    #         del sub3_dataloader.sample_dict
    #
    #         del mas_dataloader
    #         del sub1_dataloader
    #         del sub2_dataloader
    #         del sub3_dataloader
    #
    #         del mas_renderer
    #         del sub1_renderer
    #         del sub2_renderer
    #         del sub3_renderer
    #
    #         del model
    #         del model_obj
    #         del loss_func
    #
    #         torch.cuda.empty_cache()
    #
    #
    # print("total processed time(min) : ", round((time.time() - t0) / 60., 2))
    # print("total processed frames : ", save_num)
    #
    # print("(fill in google sheets) unvalid trials with wrong object pose data : ", obj_unvalid_trials)

if __name__ == "__main__":
    app.run(main)


