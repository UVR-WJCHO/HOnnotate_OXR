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
import traceback


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
flags.DEFINE_string('db', 'interaction_data', 'target db Name')   ## name ,default, help
# flags.DEFINE_string('db_source', 'source_data', 'target db Name')   ## name ,default, help
# flags.DEFINE_string('db_output', 'label_data', 'target db Name')   ## name ,default, help

flags.DEFINE_string('obj_db', 'object_data_all', 'obj db Name')   ## name ,default, help

camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'dataset')

subjectDir = os.path.join(baseDir, FLAGS.db, 'subject_info.xlsx')
subjects_df = pd.read_excel(subjectDir, header=0)
# print(subjects_df)

missing_subject_id = [48, 49]
logDir = os.path.join(os.getcwd(), 'log')


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
def modify_annotation(targetDir, outputDir, seq, trialName, data_len, flag_exist, tqdm_func, global_tqdm):
    with tqdm_func(total=data_len) as progress:

        ## log for error json
        error_name = "log_error_json_" + str(seq) + "_" + str(trialName) + ".txt"
        done_name = logDir + "/log_done_json_" + str(seq) + "_" + str(trialName) + ".txt"
        remove_name = logDir + "/log_removed_json_" + str(seq) + "_" + str(trialName) + ".txt"

        error_name = os.path.join(logDir, error_name)
        done_name = os.path.join(logDir, done_name)
        remove_name = os.path.join(logDir, remove_name)

        error_log = open(error_name, "w")
        done_log = open(done_name, "w")
        remove_log = open(remove_name, "w")

        try:
            progress.set_description(f"{seq} - {trialName}")
            #seq ='230822_S01_obj_01_grasp_13'
            db = seq.split('_')[0]
            subject_id = seq.split('_')[1][1:]
            obj_id = seq.split('_')[3]
            grasp_id = seq.split('_')[5]
            trial_num = trialName.split('_')[1]
            cam_list = ['mas', 'sub1', 'sub2', 'sub3']

            anno_base_path = os.path.join(outputDir, seq, trialName, 'annotation')
            vis_base_path = os.path.join(outputDir, seq, trialName, 'visualization')
            log_base_path = os.path.join(outputDir, seq, trialName, 'log')

            depth_base_path = os.path.join(targetDir, seq, trialName, 'depth')
            rgb_base_path = os.path.join(targetDir, seq, trialName, 'rgb')

            anno_list_4cam = []
            new_cam_list = []
            for flag_ex, cam in zip(flag_exist, cam_list):
                if flag_ex:
                    anno_list = os.listdir(os.path.join(anno_base_path, cam))
                    anno_list = natsorted(anno_list)

                    anno_list_4cam.append(anno_list)
                    new_cam_list.append(cam)

            calib_error = CAM_calibration_error[int(db)]
            calib_error = "{:.4f}".format(calib_error)

            obj_dir_name = "_".join(seq.split('_')[:-2])
            obj_pose_dir = os.path.join(baseDir, FLAGS.obj_db, db + '_obj', obj_dir_name)
            obj_data_name = obj_dir_name + '_inter_' + str(grasp_id) + '_' + str("%02d" % int(trial_num))



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
            output_num = int(origin_num / 2)
            # output_num 78
            skip_num = 3
            # if output_num * 2 <= origin_num:
            #     skip_num = 2
            # if output_num * 3 <= origin_num:
            #     skip_num = 3

            default_marker_pose = [[1.3158170e+02,   1.8510995e+01,   2.0597690e+02],  [1.2638716e+02,   1.7263596e+02,   2.0630109e+02],   [1.1576410e+02,   1.7162572e+02,  -2.3475454e+00]]
            marker_cam_pose = {}
            marker_cam_pose['marker_num'] = obj_data['marker_num']
            for idx in range(output_num):
                marker_cam_pose[str(idx)] = obj_data[str(idx*skip_num)]


            ### update log with 2D tip ###
            # tip_db_dir = os.path.join(baseDir, FLAGS.tip_db, seq, trialName)

            device = 'cuda'
            mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
            mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                                   center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

            ## load each camera parameters ##
            Ks_list = []
            Ms_list = []
            for camIdx, camID in enumerate(new_cam_list):
                anno_first = anno_list_4cam[camIdx][0]
                anno_path = os.path.join(anno_base_path, camID, anno_first)
                with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
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


            ## parameter for face blur
            init_bbox = {}
            init_bbox["sub2"] = [1100, 0, 800, 200]
            init_bbox["sub3"] = [250, 0, 800, 200]    # 키 185
            # init_bbox["sub3"] = [450, 0, 600, 350]  # 키 135

            for i in range(len(anno_list_4cam[0])):
                anno_path = os.path.join(anno_base_path, new_cam_list[0], anno_list_4cam[0][i])
                ## find mano scale, xyz_root ##
                with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                    try:
                        anno = json.load(file)
                    except Exception as e:
                        trace_back = traceback.format_exc()
                        message = str(e) + "\n" + str(trace_back)
                        error_log.write(message + "\n")

                        error_log.close()
                        done_log.close()
                        remove_log.close()
                        return None

                hand_joints = anno['annotations'][0]['data']
                hand_mano_rot = anno['Mesh'][0]['mano_trans']
                hand_mano_pose = anno['Mesh'][0]['mano_pose']
                hand_mano_shape = anno['Mesh'][0]['mano_betas']

                ## if hand_mano_rot value has exact type, continue processing
                if not isinstance(hand_mano_rot, str):
                    break

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

            for camIdx, anno_list in enumerate(anno_list_4cam):
                camID = new_cam_list[camIdx]
                for anno_name in anno_list:
                    try:
                        ## found annotation file with wrong name
                        if anno_name.split('_')[0] != camID:
                            os.remove(os.path.join(anno_base_path, camID, anno_name))

                            progress.update()
                            global_tqdm.update()
                            continue

                        frame = int(anno_name.split('_')[1][:-5])
                        anno_path = os.path.join(anno_base_path, camID, anno_name)
                        with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                            try:
                                flag_error = False
                                anno = json.load(file)
                            except Exception as e:
                                trace_back = traceback.format_exc()
                                message = str(e) + "\n" + str(trace_back)
                                error_log.write(message + "\n")
                                flag_error = True
                                pass

                            if flag_error:
                                progress.update()
                                global_tqdm.update()
                                continue

                        ## if annotation is not processed, delete
                        if isinstance(anno['Mesh'][0]['mano_trans'], str):
                            anno_path = os.path.join(anno_base_path, camID, anno_name)
                            depth_path = os.path.join(depth_base_path, camID, camID + '_' + str(frame) + '.png')
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

                            remove_log.write(anno_path + "\n")
                            progress.update()
                            global_tqdm.update()
                            continue

                        ## modify json, eval 2Dtip error
                        ### load current annotation(include updated meta info.)
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
                            if str(frame) in marker_cam_pose:
                                anno['object']['markers_data'] = marker_cam_pose[str(frame)].tolist()
                            else:
                                anno['object']['markers_data'] = default_marker_pose
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

                            bbox_hand = extractBbox(keypts_cam)

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
                            with open(anno_path, 'w', encoding='UTF-8 SIG') as file:
                                json.dump(anno, file, indent='\t', ensure_ascii=False)
                        else:
                            keypts_cam = np.squeeze(np.asarray(anno['hand']["projected_2D_pose_per_cam"]))
                            bbox_hand = extractBbox(keypts_cam)

                        ## blur face (sub2, sub3)

                        if camID == 'sub2' or camID == 'sub3':
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
                            rgb_roi_hand = rgb_init[int(bbox_hand[1]):int(bbox_hand[3]), int(bbox_hand[0]):int(bbox_hand[2])].copy()

                            # blur first, then non-blur hand region
                            rgb_init[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = blurred_roi
                            rgb_init[int(bbox_hand[1]):int(bbox_hand[3]), int(bbox_hand[0]):int(bbox_hand[2])] = rgb_roi_hand

                            # cv2.imshow("a ", rgb_init)
                            # cv2.waitKey(0)
                            cv2.imwrite(rgb_path, rgb_init)

                            del rgb_init
                            del blurred_roi
                            del rgb_roi_hand

                        progress.update()
                        global_tqdm.update()

                        done_log.write(anno_path + "\n")

                    except Exception as e:
                        trace_back = traceback.format_exc()
                        message = str(e) + "\n" + str(trace_back)
                        error_log.write(message + "\n")
                        pass
            del hand_mano_rot
            del hand_mano_pose
            del hand_mano_shape
            del mano_param

        except Exception as e:
            trace_back = traceback.format_exc()
            message = str(e) + "\n" + str(trace_back)
            error_log.write(message + "\n")
            pass

        error_log.close()
        done_log.close()
        remove_log.close()

    return True



def error_callback(result):
    print("Error!")
    pass

def done_callback(result):
    # print("Done. Result: ", result)
    pass


def main():
    try:
        t1 = time.time()
        process_count = 4
        tasks = []
        total_count = 0

        rootDir = os.path.join(baseDir, FLAGS.db)#, FLAGS.db_source)
        outputDir = os.path.join(baseDir, FLAGS.db)#, FLAGS.db_output)

        seq_list = natsorted(os.listdir(outputDir))

        os.makedirs(logDir, exist_ok=True)

        for seqIdx, seqName in enumerate(seq_list):
            seqDir = os.path.join(outputDir, seqName)

            for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
                data_len = 0
                flag_exist = []

                vis_path = os.path.join(outputDir, seqName, trialName, 'visualization')
                if not os.path.exists(vis_path):
                    shutil.rmtree(os.path.join(outputDir, seqName, trialName))
                    continue

                mas_path = os.path.join(outputDir, seqName, trialName, 'annotation', 'mas')
                if os.path.exists(mas_path):
                    data_len_0 = len(os.listdir(mas_path))
                    total_count += data_len_0
                    data_len += data_len_0
                    flag_exist.append(True)
                else:
                    flag_exist.append(False)

                sub1_path = os.path.join(outputDir, seqName, trialName, 'annotation', 'sub1')
                if os.path.exists(sub1_path):
                    data_len_1 = len(os.listdir(sub1_path))
                    total_count += data_len_1
                    data_len += data_len_1
                    flag_exist.append(True)
                else:
                    flag_exist.append(False)

                sub2_path = os.path.join(outputDir, seqName, trialName, 'annotation', 'sub2')
                if os.path.exists(sub2_path):
                    data_len_2 = len(os.listdir(sub2_path))
                    total_count += data_len_2
                    data_len += data_len_2
                    flag_exist.append(True)
                else:
                    flag_exist.append(False)

                sub3_path = os.path.join(outputDir, seqName, trialName, 'annotation', 'sub3')
                if os.path.exists(sub3_path):
                    data_len_3 = len(os.listdir(sub3_path))
                    total_count += data_len_3
                    data_len += data_len_3
                    flag_exist.append(True)
                else:
                    flag_exist.append(False)

                tasks.append((modify_annotation, (rootDir, outputDir, seqName, trialName, data_len,flag_exist,)))

        pool = TqdmMultiProcessPool(process_count)
        with tqdm.tqdm(total=total_count) as global_tqdm:
            pool.map(global_tqdm, tasks, error_callback, done_callback)


        print("---------------end postprocess ---------------")
        proc_time = round((time.time() - t1) / 60., 2)
        print("total process time : %s min" % (str(proc_time)))
    except KeyboardInterrupt:
        pass



if __name__ == '__main__':
    main()