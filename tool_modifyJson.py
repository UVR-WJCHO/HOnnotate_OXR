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

from manopth.manolayer import ManoLayer


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'test_result', 'target db Name')   ## name ,default, help
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'dataset')



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

    anno_base_path = os.path.join(targetDir, seq, trialName, 'annotation')

    frame_num = len(os.listdir(os.path.join(anno_base_path, 'mas')))

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
        Ks = torch.FloatTensor([[Ks[0], 0, Ks[2]], [0, Ks[1], Ks[3]], [0, 0, 1]]).to(device)
        Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
        Ms = torch.Tensor(np.reshape(Ms, (3, 4))).to(device)
        Ks_list.append(Ks)
        Ms_list.append(Ms)


    for frame in range(frame_num):
        anno_path_list = []
        for camID in cam_list:
            anno_path_list.append(os.path.join(anno_base_path, camID ,f'anno_{frame:04}.json'))


        ## find mano scale, xyz_root ##
        anno_path = anno_path_list[0]
        with open(anno_path, 'r', encoding='cp949') as file:
            anno = json.load(file)

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
            anno['actor'] = None

            ## calibration error 적용

            ## 데이터 scale 통일
            obj_mat = np.asarray(anno['Mesh'][0]['object_mat'])
            obj_mat[:3, -1] *= 0.1
            anno['Mesh'][0]['object_mat'] = obj_mat.tolist()

            ## image-file_name list 두개인지 체크
            
            ## contact 라벨 구조 위치 확인


            ### Addition on Meta data ###
            anno['meta']['mano_scale'] = scale
            anno['meta']['mano_xyzroot'] = xyz_root

            joints = torch.FloatTensor(anno['annotations'][0]['data']).reshape(21, 3)
            joints_cam = torch.unsqueeze(mano3DToCam3D(joints, Ms_list[camIdx]), 0)
            keypts_cam = projectPoints(joints_cam, Ks_list[camIdx])
            anno['meta']["3D_pose_per_cam"] = np.squeeze(np.asarray(joints_cam)).tolist()
            anno['meta']["projected_2D_pose_per_cam"] = np.squeeze(np.asarray(keypts_cam)).tolist()


            ### save modified annotation
            with open(anno_path, 'w', encoding='cp949') as file:
                json.dump(anno, file, indent='\t', ensure_ascii=False)


def main():
    rootDir = os.path.join(baseDir, FLAGS.db)
    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(rootDir, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            modify_annotation(rootDir, seqName, trialName)



if __name__ == '__main__':
    main()