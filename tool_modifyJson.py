import os
import sys
import pickle
from absl import flags
from absl import app
import json
from natsort import natsorted
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from manopth.manolayer import ManoLayer
from tqdm import tqdm
from config import *


"""
config.py
tool_modifyJson.py
subject_info.xlsx
dataset
--- test_result
------ 230923_S34_obj_01_grasp_19
--------- trial_0
------------ annotation
------------ depth
------------ rgb
------------ visualization
--------- trial_1
------------ annotation
------------ depth
------------ rgb
------------ visualization

--- test_object_data_all
------ 230923_obj
--------- 230923_S34_obj_01
------------ 230923_S34_obj_01_grasp_13_00.txt


"""


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'test_result', 'target db Name')   ## name ,default, help
flags.DEFINE_string('obj_db', 'test_obj_data_all', 'obj db Name')   ## name ,default, help

camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'dataset')

subjects_df = pd.read_excel('./subject_info.xlsx', header=0)
# print(subjects_df)


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

# targetDir = 'dataset/FLAGS.db'
# seq = '230923_S34_obj_01_grasp_13'
# trialName = 'trial_0'
def modify_annotation(targetDir, seq, trialName):
    #seq ='230822_S01_obj_01_grasp_13'
    db = seq.split('_')[0]
    subject_id = seq.split('_')[1][1:]
    obj_id = seq.split('_')[3]
    grasp_id = seq.split('_')[5]
    trial_num = trialName.split('_')[1]
    cam_list = ['mas', 'sub1', 'sub2', 'sub3']

    calib_error = CAM_calibration_error[int(db)]
    calib_error = "{:.4f}".format(calib_error)

    obj_dir_name = "_".join(seq.split('_')[:-2])
    obj_pose_dir = os.path.join(baseDir, FLAGS.obj_db, db + '_obj', obj_dir_name)
    obj_data_name = obj_dir_name + '_grasp_' + str("%02d" % int(grasp_id)) + '_' + str("%02d" % int(trial_num))
    marker_cam_data_name = obj_data_name + '_marker.pkl'
    marker_cam_data_path = os.path.join(obj_pose_dir, marker_cam_data_name)
    with open(marker_cam_data_path, 'rb') as f:
        marker_cam_pose = pickle.load(f)

    anno_base_path = os.path.join(targetDir, seq, trialName, 'annotation')
    depth_base_path = os.path.join(targetDir, seq, trialName, 'depth')
    rgb_base_path = os.path.join(targetDir, seq, trialName, 'rgb')
    vis_base_path = os.path.join(targetDir, seq, trialName, 'visualization')

    anno_list = os.listdir(os.path.join(anno_base_path, 'mas'))

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

        Ks = np.asarray(str(anno['calibration']['intrinsic']).split(','), dtype=float)
        Ks = torch.FloatTensor([[Ks[0], 0, Ks[2]], [0, Ks[1], Ks[3]], [0, 0, 1]])
        Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
        Ms = torch.Tensor(np.reshape(Ms, (3, 4)))
        Ks_list.append(Ks)
        Ms_list.append(Ms)


    for anno_name in tqdm(anno_list):
        frame = int(anno_name[5:9])

        anno_path_list = []
        for camID in cam_list:
            anno_path_list.append(os.path.join(anno_base_path, camID, anno_name))

        ## find mano scale, xyz_root ##
        anno_path = anno_path_list[0]
        with open(anno_path, 'r', encoding='cp949') as file:
            anno = json.load(file)

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


        for camIdx, anno_path in enumerate(anno_path_list):
            ### load current annotation(include updated meta info.)
            with open(anno_path, 'r', encoding='cp949') as file:
                anno = json.load(file)


            ## 피험자 정보 적용
            data_for_id = get_data_by_id(subjects_df, subject_id)

            age = str(data_for_id['나이대'])
            sex_str = data_for_id['성별']
            if sex_str == '남':
                sex = "M"
            else:
                sex = "F"
            height = str(data_for_id['키'])
            size = str(data_for_id['손크기'])

            anno['actor']['id'] = age
            anno['actor']['sex'] = sex
            anno['actor']['height'] = height
            anno['actor']['handsize'] = size

            ## object marker 데이터 추가
            anno['object']['marker_count'] = marker_cam_pose['marker_num']
            anno['object']['markers_data'] = marker_cam_pose[str(frame)].tolist()
            anno['object']['pose_data'] = anno['Mesh'][0]['object_mat']

            ## calibration error 적용
            anno['calibration']['error'] = float(calib_error)

            ## 데이터 scale 통일
            obj_mat = np.asarray(anno['Mesh'][0]['object_mat'])
            obj_mat[:3, -1] *= 0.1
            anno['Mesh'][0]['object_mat'] = obj_mat.tolist()


            ### Addition on Meta data ###
            anno['hand'] = {}
            anno['hand']['mano_scale'] = scale
            anno['hand']['mano_xyz_root'] = xyz_root.tolist()

            joints = torch.FloatTensor(anno['annotations'][0]['data']).reshape(21, 3)
            joints_cam = torch.unsqueeze(mano3DToCam3D(joints, Ms_list[camIdx]), 0)
            keypts_cam = projectPoints(joints_cam, Ks_list[camIdx])
            anno['hand']["3D_pose_per_cam"] = np.squeeze(np.asarray(joints_cam)).tolist()
            anno['hand']["projected_2D_pose_per_cam"] = np.squeeze(np.asarray(keypts_cam)).tolist()

            obj_mat = torch.FloatTensor(np.asarray(anno['Mesh'][0]['object_mat']))
            obj_mat_cam = obj_mat.T @ Ms_list[camIdx].T
            obj_mat_cam = np.squeeze(np.asarray(obj_mat_cam.T)).tolist()
            obj_mat_cam.append([0.0, 0.0, 0.0, 1.0])
            anno['object']["6D_pose_per_cam"] = obj_mat_cam


            ## image-file_name list 두개인지 체크(덮어씌우기)
            camID = camIDset[camIdx]
            anno['images']['file_name'] = [os.path.join("rgb", str(camID), str(camID) + '_' + str(frame) + '.jpg'), os.path.join("depth", str(camID), str(camID) + '_' + str(frame) + '.png')]

            ## contact 라벨 구조 위치 확인
            if 'contact' in anno['Mesh'][0]:
                contact = anno['Mesh'][0]['contact']
                anno['contact'] = contact
                del anno['Mesh'][0]['contact']

            ### save modified annotation
            with open(anno_path, 'w', encoding='cp949') as file:
                json.dump(anno, file, indent='\t', ensure_ascii=False)


def main():
    rootDir = os.path.join(baseDir, FLAGS.db)
    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(rootDir, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            print("start processing %s - %s" % (seqName, trialName))
            modify_annotation(rootDir, seqName, trialName)



if __name__ == '__main__':
    main()