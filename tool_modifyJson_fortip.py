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


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '20231207', 'target db Name')   ## name ,default, help
flags.DEFINE_string('db_source', 'source_data', 'target db Name')   ## name ,default, help
flags.DEFINE_string('db_output', 'label_data', 'target db Name')   ## name ,default, help

flags.DEFINE_string('obj_db', 'obj_data_all', 'obj db Name')   ## name ,default, help
flags.DEFINE_string('tip_db', 'handtip', 'tip db Name')   ## name ,default, help

camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'data', 'NIA_output', 'HAND_GESTURE')

# subjects_df = pd.read_excel('./subject_info.xlsx', header=0)
# print(subjects_df)

missing_subject_id = [48, 49]

error_log = open("log_error_json_list_tip.txt", "w")
error_tip_log = open("log_wrong_tip_list.txt", "w")

# targetDir = 'dataset/FLAGS.db'
# seq = '230923_S34_obj_01_grasp_13'
# trialName = 'trial_0'
def modify_annotation(targetDir, outputDir, seq, trialName, data_len, flag_exist, tqdm_func, global_tqdm):
    with tqdm_func(total=data_len) as progress:
        progress.set_description(f"{seq} - {trialName}")

        #seq ='230822_S01_obj_01_grasp_13'
        db = seq.split('_')[0]
        subject_id = seq.split('_')[1][1:]
        obj_id = seq.split('_')[3]
        grasp_id = seq.split('_')[5]
        trial_num = trialName.split('_')[1]
        cam_list = ['mas', 'sub1', 'sub2', 'sub3']

        anno_base_path = os.path.join(outputDir, seq, trialName, 'annotation')
        log_base_path = os.path.join(outputDir, seq, trialName, 'log')
        assert os.path.exists(log_base_path), "No log folder in "+log_base_path

        ### update log with 2D tip ###
        tip_db_dir = os.path.join(outputDir, seq, trialName, FLAGS.tip_db)
        # vis_db_dir = os.path.join(outputDir, seq, trialName, 'rgb')

        anno_list_4cam = []
        new_cam_list = []
        for flag_ex, cam in zip(flag_exist, cam_list):
            if flag_ex:
                anno_list = os.listdir(os.path.join(anno_base_path, cam))
                anno_list_4cam.append(anno_list)
                new_cam_list.append(cam)

        ## load each camera parameters ##
        # Ks_list = []
        # Ms_list = []
        # for camIdx, camID in enumerate(new_cam_list):
        #     anno_first = anno_list_4cam[camIdx][0]
        #     anno_path = os.path.join(anno_base_path, camID, anno_first)
        #     with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
        #         anno = json.load(file)
        #
        #     Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
        #     Ms = np.reshape(Ms, (3, 4))
        #     if isinstance(anno['calibration']['intrinsic'], str):
        #         Ks = np.asarray(str(anno['calibration']['intrinsic']).split(','), dtype=float)
        #         Ks = torch.FloatTensor([[Ks[0], 0, Ks[2]], [0, Ks[1], Ks[3]], [0, 0, 1]])
        #         Ms[:, -1] = Ms[:, -1] / 10.0
        #     else:
        #         Ks = torch.FloatTensor(np.squeeze(np.asarray(anno['calibration']['intrinsic'])))
        #
        #     Ms = torch.Tensor(Ms)
        #     Ks_list.append(Ks)
        #     Ms_list.append(Ms)

        dfs = {}
        dfs['tip_precision'] = pd.DataFrame([], columns=['avg'])
        dfs['tip_recall'] = pd.DataFrame([], columns=['avg'])
        dfs['tip_f1'] = pd.DataFrame([], columns=['avg'])
        total_metrics = [0.] * 3

        TP = 1e-7  # 각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이내
        FP = 1e-7  # 각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이상
        FN = 1e-7  # 참값이 존재하지만 키포인트 좌표가 존재하지 않는 경우(미태깅)
        for camIdx, anno_list in enumerate(anno_list_4cam):
            camID = new_cam_list[camIdx]

            for anno_name in anno_list:
                frame = int(anno_name.split('_')[1][:-5])
                anno_path = os.path.join(anno_base_path, camID, anno_name)

                # manual 2D tip GT
                tip_data_name = str(camID) + '_' + str(frame) + '.json'
                tip_data_path = os.path.join(tip_db_dir, camID, tip_data_name)

                if os.path.exists(tip_data_path):
                    with open(tip_data_path, "r", encoding='UTF-8 SIG') as data:
                        try:
                            flag_error = False
                            tip_data = json.load(data)['annotations'][0]
                        except:
                            error_log.write(tip_data_path + '\n')
                            flag_error = True
                            pass
                        if flag_error:
                            progress.update()
                            global_tqdm.update()
                            continue

                    tip_kpts = {}
                    for tip in tip_data:
                        valid = tip['status']
                        if valid:
                            tip_name = tip['label']
                            tip_2d = [tip['x'], tip['y']]
                            tip_kpts[tip_name] = np.round(tip_2d, 2)

                    tip2d_np = []
                    tip2d_idx = []
                    for key in tip_kpts.keys():
                        tip2d_np.append(tip_kpts[key])
                        tip2d_idx.append(CFG_TIP_IDX[key])

                    with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                        try:
                            flag_error = False
                            anno = json.load(file)
                        except:
                            error_log.write(anno_path + '\n')
                            flag_error = True
                            pass
                        if flag_error:
                            progress.update()
                            global_tqdm.update()
                            continue

                    keypts_cam = np.squeeze(np.asarray(anno['hand']["projected_2D_pose_per_cam"]))

                    # our dataset
                    proj_2Dkpts = np.squeeze(np.asarray(keypts_cam))
                    proj_2Dkpts = proj_2Dkpts[tip2d_idx, :]

                    # calculate 2D keypoints F1-Score for each view
                    for idx, gt_kpt in enumerate(tip2d_np):
                        pred_kpt = proj_2Dkpts[idx]
                        if (pred_kpt == None).any():
                            FN += 1
                        dist = np.linalg.norm(gt_kpt - pred_kpt)
                        if dist < 50:
                            TP += 1
                        elif dist >= 50:
                            FP += 1

                    global_tqdm.update()
                    progress.update()

        keypoint_precision_score = TP / (TP + FP)
        keypoint_recall_score = TP / (TP + FN)
        keypoint_f1_score = 2 * (keypoint_precision_score * keypoint_recall_score /
                                 (keypoint_precision_score + keypoint_recall_score))  # 2*TP/(2*TP+FP+FN)

        total_metrics[0] += keypoint_precision_score
        total_metrics[1] += keypoint_recall_score
        total_metrics[2] += keypoint_f1_score

        if keypoint_precision_score != 0.0 and keypoint_precision_score != 0.5:
            ## save tipGT keypoint error (remove original mediapipe kpts error file, check it.
            csv_files = ['tip_precision', 'tip_recall', 'tip_f1']
            for idx, file in enumerate(csv_files):
                with open(os.path.join(log_base_path, file + '.csv'), "w", encoding='UTF-8 SIG') as f:
                    ws = csv.writer(f)
                    ws.writerow(['total_avg', total_metrics[idx]])
                    ws.writerow(['camID', 'avg'])
                df = dfs[file]
                df.to_csv(os.path.join(log_base_path, file + '.csv'), mode='a', index=True, header=False)

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

    rootDir = os.path.join(baseDir, FLAGS.db, FLAGS.db_source)
    outputDir = os.path.join(baseDir, FLAGS.db, FLAGS.db_output)

    seq_list = natsorted(os.listdir(outputDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(outputDir, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            data_len = 0
            flag_exist = []
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



if __name__ == '__main__':
    main()

    error_log.close()
    error_tip_log.close()
    print("end")