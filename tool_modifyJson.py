import os
import sys
import pickle
from absl import flags, app
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import csv
import time
from natsort import natsorted

import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool

from config import *
from manopth.manolayer import ManoLayer
from utils.lossUtils import *

"""
[실행]
> python tool_modifyJson.py --db {  } --tip_db {   }

[필요 데이터]
> dataset/{FLAGS.db} : 해당 폴더 하위에 모든 시퀀스 폴더 구성 (tip, obj 데이터와의 구분을 위해 필요함)
> dataset/{FLAGS.tip_db} : 해당 폴더 하위에 수집된 tip 데이터 시퀀스 폴더 구성 (전부 없어도됨. 있는 것만 tip error 생성됨)
> dataset/object_data_all : 드랍박스에서 모든 물체 자세 정리한 데이터 받아서 구성(카이스트 공유 예정)

[디렉토리 구조 예시]

config.py
tool_modifyJson.py
subject_info.xlsx
dataset
--- {FLAGS.db}
------ 230923_S34_obj_01_grasp_19
--------- trial_0
------------ annotation
------------ depth
------------ rgb
------------ visualization

--- {FLAGS.tip_db}
------ 230923_S34_obj_01_grasp_19
--------- trial_0
------------ mas
--------------- mas_0.json
--------------- mas_1.json

--- object_data_all(드랍박스 링크로 공유 예정)
------ 230923_obj
--------- 230923_S34_obj_01
------------ 230923_S34_obj_01_grasp_13_00.txt

"""


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'test_result', 'target db Name')   ## name ,default, help
flags.DEFINE_string('obj_db', 'obj_data_all', 'obj db Name')   ## name ,default, help
flags.DEFINE_string('tip_db', 'test_result_tip', 'tip db Name')   ## name ,default, help

camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'dataset')

subjects_df = pd.read_excel('./subject_info.xlsx', header=0)
# print(subjects_df)

missing_subject_id = [48, 49]


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


def get_data_by_id(df, id_value):
    row_data = df.loc[int(id_value)-1, :]
    assert int(row_data['Subject']) == int(id_value), "subject mismatch. check excel file"

    if not row_data.empty:
        return row_data.to_dict()
    else:
        return "No data found for the given ID."


def extractBbox(proj_kpts2D, image_rows=1080, image_cols=1920, w_margin=300, h_margin=150):
        # consider fixed size bbox
        x_min = min(proj_kpts2D[:, 0])
        x_max = max(proj_kpts2D[:, 0])
        y_min = min(proj_kpts2D[:, 1])
        y_max = max(proj_kpts2D[:, 1])

        x_min = max(0, x_min - w_margin)
        y_min = max(0, y_min - h_margin)

        if (x_max + w_margin) > image_cols:
            x_max = image_cols
        else:
            x_max = x_max + w_margin

        if (y_max + h_margin) > image_rows:
            y_max = image_rows
        else:
            y_max = y_max + h_margin

        bbox = [x_min, y_min, x_max, y_max]
        return bbox


# targetDir = 'dataset/FLAGS.db'
# seq = '230923_S34_obj_01_grasp_13'
# trialName = 'trial_0'
def modify_annotation(targetDir, seq, trialName, data_len, tqdm_func, global_tqdm):
    with tqdm_func(total=data_len) as progress:
        progress.set_description(f"{seq} - {trialName}")

        #seq ='230822_S01_obj_01_grasp_13'
        db = seq.split('_')[0]
        subject_id = seq.split('_')[1][1:]
        obj_id = seq.split('_')[3]
        grasp_id = seq.split('_')[5]
        trial_num = trialName.split('_')[1]
        cam_list = ['mas', 'sub1', 'sub2', 'sub3']


        anno_base_path = os.path.join(targetDir, seq, trialName, 'annotation')
        depth_base_path = os.path.join(targetDir, seq, trialName, 'depth')
        rgb_base_path = os.path.join(targetDir, seq, trialName, 'rgb')
        vis_base_path = os.path.join(targetDir, seq, trialName, 'visualization')
        anno_list = os.listdir(os.path.join(anno_base_path, 'mas'))

        output_num = len(anno_list)

        calib_error = CAM_calibration_error[int(db)]
        calib_error = "{:.4f}".format(calib_error)

        obj_dir_name = "_".join(seq.split('_')[:-2])
        obj_pose_dir = os.path.join(baseDir, FLAGS.obj_db, db + '_obj', obj_dir_name)
        obj_data_name = obj_dir_name + '_grasp_' + str("%02d" % int(grasp_id)) + '_' + str("%02d" % int(trial_num))



        # marker_cam_data_name = obj_data_name + '_marker.pkl'
        # marker_cam_data_path = os.path.join(obj_pose_dir, marker_cam_data_name)
        #
        # with open(marker_cam_data_path, 'rb') as f:
        #     marker_cam_pose = pickle.load(f)

        obj_pose_data = os.path.join(obj_pose_dir, obj_data_name + '.txt')

        obj_data = {}
        with open(obj_pose_data, "r") as f:
            line = f.readline().strip().split(' ')
            _ = f.readline()
            marker_num = int(float(line[0]))

            obj_data['marker_num'] = marker_num
            frame = 0
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.split(' ')
                line = [value for value in line if value != '']
                marker_pose = np.zeros((marker_num, 3))
                for i in range(marker_num):
                    marker_pose[i, 0] = float(line[i * 3 + 1])
                    marker_pose[i, 1] = float(line[i * 3 + 2])
                    marker_pose[i, 2] = float(line[i * 3 + 3])
                obj_data[str(frame)] = marker_pose
                frame += 1

        origin_num = len(obj_data) - 1  # 78 * 3 or 78* 2
        # output_num 78
        if output_num * 2 <= origin_num:
            skip_num = 2
        if output_num * 3 <= origin_num:
            skip_num = 3

        marker_cam_pose = {}
        marker_cam_pose['marker_num'] = obj_data['marker_num']
        for idx in range(output_num):
            if idx % skip_num != 0:
                continue
            marker_cam_pose[str(idx)] = obj_data[str(idx*skip_num)]


        ### update log with 2D tip ###
        tip_db_dir = os.path.join(baseDir, FLAGS.tip_db, seq, trialName)
        log_base_path = os.path.join(targetDir, seq, trialName, 'log')

        device = 'cuda'
        mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
        mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                               center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

        ## load each camera parameters ##
        Ks_list = []
        Ms_list = []
        for camID in cam_list:
            anno_path = os.path.join(anno_base_path, camID, 'anno_0000.json')
            with open(anno_path, 'r', encoding='cp949') as file:
                anno = json.load(file)

            if isinstance(anno['calibration']['intrinsic'], str):
                Ks = np.asarray(str(anno['calibration']['intrinsic']).split(','), dtype=float)
                Ks = torch.FloatTensor([[Ks[0], 0, Ks[2]], [0, Ks[1], Ks[3]], [0, 0, 1]])
            else:
                Ks = torch.FloatTensor(np.squeeze(np.asarray(anno['calibration']['intrinsic'])))
            Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
            Ms = np.reshape(Ms, (3, 4))
            Ms[:, -1] = Ms[:, -1] / 10.0
            Ms = torch.Tensor(Ms)

            Ks_list.append(Ks)
            Ms_list.append(Ms)

        dfs = {}
        dfs['tip_precision'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        dfs['tip_recall'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        dfs['tip_f1'] = pd.DataFrame([], columns=['mas', 'sub1', 'sub2', 'sub3', 'avg'])
        total_metrics = [0.] * 3
        eval_num = 0

        ## parameter for face blur
        init_bbox = {}
        init_bbox["sub2"] = [1200, 0, 700, 250]
        init_bbox["sub3"] = [450, 0, 600, 250]    # 키 185
        # init_bbox["sub3"] = [450, 0, 600, 350]  # 키 135

        for anno_name in anno_list:
            frame = int(anno_name[5:9])

            anno_path_list = []
            for camID in cam_list:
                anno_path_list.append(os.path.join(anno_base_path, camID, anno_name))

            ## find mano scale, xyz_root ##
            anno_path = anno_path_list[0]
            with open(anno_path, 'r', encoding='cp949') as file:
                anno = json.load(file)

            ## if annotation is not processed, delete
            if isinstance(anno['annotations'][0]['data'], str):
                print("unprocess json : %s, remove json" % anno_name)
                for camID in cam_list:
                    anno_path = os.path.join(anno_base_path, camID, anno_name)
                    depth_path = os.path.join(depth_base_path, camID, camID+'_' + str(frame)+'.png')
                    rgb_path = os.path.join(rgb_base_path, camID, camID + '_' + str(frame) + '.jpg')
                    vis_path = os.path.join(vis_base_path, camID, 'blend_pred_' + camID + '_' + str(frame) + '.png')
                    if os.path.isfile(anno_path):
                        os.remove(anno_path)
                    if os.path.isfile(depth_path):
                        os.remove(depth_path)
                    if os.path.isfile(rgb_path):
                        os.remove(rgb_path)
                    if os.path.isfile(vis_path):
                        os.remove(vis_path)
                continue

            hand_joints = anno['annotations'][0]['data']
            hand_mano_rot = anno['Mesh'][0]['mano_trans']
            hand_mano_pose = anno['Mesh'][0]['mano_pose']
            hand_mano_shape = anno['Mesh'][0]['mano_betas']

            hand_mano_rot = torch.FloatTensor(np.asarray(hand_mano_rot)).to(device)
            hand_mano_pose = torch.FloatTensor(np.asarray(hand_mano_pose)).to(device)
            hand_mano_shape = torch.FloatTensor(np.asarray(hand_mano_shape)).to(device)

            mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1)
            mano_verts, mano_joints = mano_layer(mano_param, hand_mano_shape)
            mano_joints = np.squeeze(mano_joints.detach().cpu().numpy())

            hand_joints = np.squeeze(np.asarray(hand_joints))

            xyz_root = hand_joints[0, :]

            hand_joints_norm = hand_joints - xyz_root
            dist_anno = hand_joints_norm[1, :] - hand_joints_norm[0, :]
            dist_mano = mano_joints[1, :] - mano_joints[0, :]

            scale = np.average(dist_mano / dist_anno)

            kpts_precision = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
            kpts_recall = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
            kpts_f1 = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}

            bbox_list = []
            ## modify json, eval 2Dtip error
            for camIdx, anno_path in enumerate(anno_path_list):
                camID = camIDset[camIdx]

                ### load current annotation(include updated meta info.)
                with open(anno_path, 'r', encoding='cp949') as file:
                    anno = json.load(file)

                ## if current annotation is postprocessed, pass
                if not isinstance(anno['calibration']['intrinsic'], list):
                    ## 피험자 정보 적용
                    data_for_id = get_data_by_id(subjects_df, subject_id)

                    age = str(data_for_id['나이대'])
                    sex_str = data_for_id['성별']
                    if sex_str == '남':
                        sex = "M"
                    else:
                        sex = "F"
                    height = float(data_for_id['키'])
                    size = float(data_for_id['손크기'])

                    actor_id = int(subject_id)
                    if actor_id in [101, 102]:
                        idx = [101, 102].index(actor_id)
                        actor_id = missing_subject_id[idx]

                    anno['actor']['id'] = 'S'+str(actor_id)
                    anno['actor']['age'] = age
                    anno['actor']['sex'] = sex
                    anno['actor']['height'] = height
                    anno['actor']['handsize'] = size


                    ## object marker 데이터 추가

                    anno['object']['marker_count'] = int(marker_cam_pose['marker_num'])
                    anno['object']['markers_data'] = marker_cam_pose[str(frame)].tolist()
                    anno['object']['pose_data'] = anno['Mesh'][0]['object_mat']

                    ## calibration error 적용
                    anno['calibration']['error'] = float(calib_error)

                    ## 데이터 scale 통일
                    obj_mat = np.asarray(anno['Mesh'][0]['object_mat'])
                    obj_mat[:3, -1] *= 0.1
                    anno['Mesh'][0]['object_mat'] = obj_mat.tolist()

                    Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
                    Ms = np.reshape(Ms, (3, 4))
                    anno['calibration']['extrinsic'] = Ms.tolist()

                    ### Addition on Meta data ###
                    anno['hand'] = {}
                    anno['hand']['mano_scale'] = scale
                    anno['hand']['mano_xyz_root'] = xyz_root.tolist()

                    joints = torch.FloatTensor(anno['annotations'][0]['data']).reshape(21, 3)
                    joints_cam = torch.unsqueeze(mano3DToCam3D(joints, Ms_list[camIdx]), 0)
                    keypts_cam = projectPoints(joints_cam, Ks_list[camIdx])

                    anno['hand']["3D_pose_per_cam"] = np.squeeze(np.asarray(joints_cam)).tolist()
                    anno['hand']["projected_2D_pose_per_cam"] = np.squeeze(np.asarray(keypts_cam)).tolist()

                    keypts_cam = np.squeeze(keypts_cam.detach().numpy())
                    ## debug
                    # rgb_path = os.path.join(rgb_base_path, camID, camID + '_' + str(frame) + '.jpg')
                    # rgb = cv2.imread(rgb_path)
                    # rgb_2d_pred = paint_kpts(None, rgb, keypts_cam)
                    # cv2.imshow("check 2d", rgb_2d_pred)
                    # cv2.waitKey(0)

                    bbox = extractBbox(keypts_cam)
                    bbox_list.append(bbox)

                    obj_mat = torch.FloatTensor(np.asarray(anno['Mesh'][0]['object_mat']))
                    obj_mat_cam = obj_mat.T @ Ms_list[camIdx].T
                    obj_mat_cam = np.squeeze(np.asarray(obj_mat_cam.T)).tolist()
                    obj_mat_cam.append([0.0, 0.0, 0.0, 1.0])
                    anno['object']["6D_pose_per_cam"] = obj_mat_cam


                    ## image-file_name list 두개인지 체크(덮어씌우기)
                    anno['images']['file_name'] = [os.path.join("rgb", str(camID), str(camID) + '_' + str(frame) + '.jpg'), os.path.join("depth", str(camID), str(camID) + '_' + str(frame) + '.png')]

                    ## contact 라벨 구조 위치 확인
                    if 'contact' in anno['Mesh'][0]:
                        contact = anno['Mesh'][0]['contact']
                        anno['contact'] = contact
                        del anno['Mesh'][0]['contact']

                    ## 기타 json 구조 수정(231117 피드백)
                    anno['kinect_camera']['id'] = int(anno['kinect_camera']['id'])
                    anno['kinect_camera']['height'] = int(anno['kinect_camera']['height'])
                    anno['kinect_camera']['width'] = int(anno['kinect_camera']['width'])

                    anno['infrared_camera'][0]['id'] = int(anno['infrared_camera'][0]['id'])
                    anno['infrared_camera'][0]['height'] = int(anno['infrared_camera'][0]['height'])
                    anno['infrared_camera'][0]['width'] = int(anno['infrared_camera'][0]['width'])
                    anno['infrared_camera'][0]['frame'] = int(anno['infrared_camera'][0]['frame'])
                    anno['infrared_camera'][0]['resolution'] = float(anno['infrared_camera'][0]['resolution'])

                    Ks = np.asarray(str(anno['calibration']['intrinsic']).split(','), dtype=float)
                    Ks = np.array([[Ks[0], 0, Ks[2]], [0, Ks[1], Ks[3]], [0, 0, 1]])
                    anno['calibration']['intrinsic'] = Ks.tolist()

                    anno['annotations'][0]['class_id'] = int(anno['annotations'][0]['class_id'])
                    anno['Mesh'][0]['class_id'] = int(anno['Mesh'][0]['class_id'])

                    ### save modified annotation
                    with open(anno_path, 'w', encoding='cp949') as file:
                        json.dump(anno, file, indent='\t', ensure_ascii=False)
                else:
                    keypts_cam = np.squeeze(np.asarray(anno['hand']["projected_2D_pose_per_cam"]))
                    bbox = extractBbox(keypts_cam)
                    bbox_list.append(bbox)

                ### generate new log data with 2D tip ###
                # manual 2D tip GT
                tip_data_dir = os.path.join(tip_db_dir, camID)
                tip_data_name = str(camID) + '_' + str(frame) + '.json'
                tip_data_path = os.path.join(tip_data_dir, tip_data_name)

                # calculate 2D tip error only if tip GT exists
                if os.path.exists(tip_data_path):
                    with open(tip_data_path, "r") as data:
                        tip_data = json.load(data)['annotations'][0]
                    tip_kpts = {}
                    for tip in tip_data:
                        tip_name = tip['label']
                        tip_2d = [tip['x'], tip['y']]
                        tip_kpts[tip_name] = np.round(tip_2d, 2)

                    tip2d_np = []
                    tip2d_idx = []
                    for key in tip_kpts.keys():
                        tip2d_np.append(tip_kpts[key])
                        tip2d_idx.append(CFG_TIP_IDX[key])

                    # our dataset
                    proj_2Dkpts = np.squeeze(np.asarray(keypts_cam))
                    proj_2Dkpts = proj_2Dkpts[tip2d_idx, :]

                    # calculate 2D keypoints F1-Score for each view
                    TP = 1e-7  # 각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이내
                    FP = 1e-7  # 각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이상
                    FN = 1e-7  # 참값이 존재하지만 키포인트 좌표가 존재하지 않는 경우(미태깅)
                    for idx, gt_kpt in enumerate(tip2d_np):
                        pred_kpt = proj_2Dkpts[idx]
                        if (pred_kpt == None).any():
                            FN += 1
                        dist = np.linalg.norm(gt_kpt - pred_kpt)
                        if dist < 50:
                            TP += 1
                        elif dist >= 50:
                            FP += 1

                    keypoint_precision_score = TP / (TP + FP)
                    keypoint_recall_score = TP / (TP + FN)
                    keypoint_f1_score = 2 * (keypoint_precision_score * keypoint_recall_score /
                                             (keypoint_precision_score + keypoint_recall_score))  # 2*TP/(2*TP+FP+FN)
                    kpts_precision[camID] = keypoint_precision_score
                    kpts_recall[camID] = keypoint_recall_score
                    kpts_f1[camID] = keypoint_f1_score

            kpts_precision_avg = sum(kpts_precision.values()) / len(cam_list)
            kpts_recall_avg = sum(kpts_recall.values()) / len(cam_list)
            kpts_f1_avg = sum(kpts_f1.values()) / len(cam_list)

            if kpts_precision_avg != 0:
                total_metrics[0] += kpts_precision_avg
                total_metrics[1] += kpts_recall_avg
                total_metrics[2] += kpts_f1_avg
                dfs['tip_precision'].loc[frame] = [kpts_precision['mas'], kpts_precision['sub1'], kpts_precision['sub2'],
                                                    kpts_precision['sub3'], kpts_precision_avg]
                dfs['tip_recall'].loc[frame] = [kpts_recall['mas'], kpts_recall['sub1'], kpts_recall['sub2'],
                                                 kpts_recall['sub3'], kpts_recall_avg]
                dfs['tip_f1'].loc[frame] = [kpts_f1['mas'], kpts_f1['sub1'], kpts_f1['sub2'], kpts_f1['sub3'],
                                             kpts_f1_avg]
                eval_num += 1

            ## blur face (sub2, sub3)
            for camIdx, camID in enumerate(camIDset):
                if camID == 'mas' or camID == 'sub1':
                    continue
                rgb_path = os.path.join(rgb_base_path, camID, camID + '_' + str(frame) + '.jpg')
                rgb_init = cv2.imread(rgb_path)

                # blur manual region
                bbox = init_bbox[camID].copy()
                if camID == 'sub3':
                    # max height : 185 cm
                    data_for_id = get_data_by_id(subjects_df, subject_id)
                    height = float(data_for_id['키'])
                    height_gap = (185 - height) * 2
                    bbox[3] = bbox[3] + height_gap

                rgb_roi = rgb_init[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                blurred_roi = cv2.GaussianBlur(rgb_roi, ksize=(0, 0), sigmaX=10, sigmaY=0)
                # non-blur hand region
                bbox_hand = bbox_list[camIdx]
                rgb_roi_hand = rgb_init[int(bbox_hand[1]):int(bbox_hand[3]), int(bbox_hand[0]):int(bbox_hand[2])].copy()

                # blur first, then non-blur hand region
                rgb_init[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = blurred_roi
                rgb_init[int(bbox_hand[1]):int(bbox_hand[3]), int(bbox_hand[0]):int(bbox_hand[2])] = rgb_roi_hand

                # cv2.imshow("a ", rgb_init)
                # cv2.waitKey(0)
                cv2.imwrite(rgb_path, rgb_init)

            progress.update()
            global_tqdm.update()

        ## save tipGT keypoint error (remove original mediapipe kpts error file, check it.
        csv_files = ['tip_precision', 'tip_recall', 'tip_f1']
        if eval_num != 0:
            for idx, file in enumerate(csv_files):
                with open(os.path.join(log_base_path, file + '.csv'), "w", encoding='utf-8') as f:
                    ws = csv.writer(f)
                    ws.writerow(['total_avg', total_metrics[idx] / eval_num])
                    ws.writerow(['frame', 'mas', 'sub1', 'sub2', 'sub3', 'avg'])
                df = dfs[file]
                df.to_csv(os.path.join(log_base_path, file + '.csv'), mode='a', index=True, header=False)
        else:
            print("no 2D tip data, skip generating csv file for tip F1-score")

    return True


def error_callback(result):
    print("Error!")

def done_callback(result):
    # print("Done. Result: ", result)
    return


def main():
    t1 = time.time()
    process_count = 4
    tasks = []
    total_count = 0

    rootDir = os.path.join(baseDir, FLAGS.db)
    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(rootDir, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            data_len = len(os.listdir(os.path.join(rootDir, seqName, trialName, 'annotation', 'mas')))
            total_count += data_len
            tasks.append((modify_annotation, (rootDir, seqName, trialName, data_len, )))

    pool = TqdmMultiProcessPool(process_count)
    with tqdm.tqdm(total=total_count) as global_tqdm:
        pool.map(global_tqdm, tasks, error_callback, done_callback)

    print("---------------end postprocess ---------------")
    proc_time = round((time.time() - t1) / 60., 2)
    print("total process time : %s min" % (str(proc_time)))



if __name__ == '__main__':
    main()
    print("end")