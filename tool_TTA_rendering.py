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
from config import *

from manopth.manolayer import ManoLayer
from modules.renderer import Renderer
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from utils.lossUtils import *


"""
1. annotation data 이용해서 visualization 생성 예시.
2. (TODO) 정량 지표 log 폴더 접근해서 전체 데이터셋에 대해 평균 목표치 만족 여부 확인
3. (TODO) 2D tip data 별도로 제공될 시 해당 데이터가 존재하는 시퀀스에 대해서는 Keypoint error 재측정 후 F1 score 생성?
"""





### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'interaction_data', 'target db Name')   ## name ,default, help
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'dataset')
objModelDir = os.path.join(baseDir, 'obj_scanned_models_230915~')   #### change to "obj_scanned_models"



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



# targetDir = 'dataset/FLAGS.db'
# seq = '230923_S34_obj_01_grasp_13'
# trialName = 'trial_0'
def load_annotation(targetDir, seq, trialName):
    #seq ='230822_S01_obj_01_grasp_13'
    db = seq.split('_')[0]
    subject_id = seq.split('_')[1][1:]
    obj_id = seq.split('_')[3]
    grasp_id = seq.split('_')[5]
    trial_num = trialName.split('_')[1]
    cam_list = ['mas', 'sub1', 'sub2', 'sub3']

    anno_base_path = os.path.join(targetDir, seq, trialName, 'annotation')
    rgb_base_path = os.path.join(targetDir, seq, trialName, 'rgb')
    depth_base_path = os.path.join(targetDir, seq, trialName, 'depth')

    frame_num = len(os.listdir(os.path.join(anno_base_path, 'mas')))

    device = 'cuda'
    mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
    mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                                center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

    ## load each camera parameters ##
    Ks_list = []
    Ms_list = []
    for camID in cam_list:
        anno_path = os.path.join(anno_base_path, camID ,'anno_0000.json')
        with open(anno_path, 'r', encoding='cp949') as file:
            anno = json.load(file)

        Ks = np.asarray(anno['calibration']['intrinsic'])
        Ks = torch.FloatTensor(Ks).to(device)
        Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
        Ms = np.reshape(Ms, (3, 4))

        # required for rendering (considering vertex scale)
        Ms[:, -1] = Ms[:, -1] / 10.0

        Ms = torch.Tensor(Ms).to(device)

        Ks_list.append(Ks)
        Ms_list.append(Ms)
    default_M = np.eye(4)[:3]

    # tensor([[942.9117,   0.0000, 964.8910],
    #         [  0.0000, 931.7506, 550.0068],
    #         [  0.0000,   0.0000,   1.0000]], device='cuda:0')
    mas_renderer = Renderer('cuda', 1, default_M, Ks_list[0], (1080, 1920))
    sub1_renderer = Renderer('cuda', 1, default_M, Ks_list[1], (1080, 1920))
    sub2_renderer = Renderer('cuda', 1, default_M, Ks_list[2], (1080, 1920))
    sub3_renderer = Renderer('cuda', 1, default_M, Ks_list[3], (1080, 1920))
    renderer_set = [mas_renderer, sub1_renderer, sub2_renderer, sub3_renderer]


    ## load hand & object template mesh ##
    hand_faces_template = mano_layer.th_faces.repeat(1, 1, 1)

    target_mesh_class = str(obj_id) + '_' + str(OBJType(int(obj_id)).name)
    obj_mesh_path = os.path.join(baseDir, objModelDir, target_mesh_class, target_mesh_class + '.obj')
    obj_scale = CFG_OBJECT_SCALE_FIXED[int(obj_id)-1]
    obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
    obj_verts_template = (obj_verts * float(obj_scale)).to(device)
    obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(device)

    h = torch.ones((obj_verts_template.shape[0], 1), device=device)
    obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

    ## per frame process ##
    for frame in range(frame_num):
        anno_path_list = []
        rgb_path_list = []
        depth_path_list = []
        for camID in cam_list:
            anno_path_list.append(os.path.join(anno_base_path, camID ,f'anno_{frame:04}.json'))
            rgb_path_list.append(os.path.join(rgb_base_path, camID, f'{camID}_{frame}.jpg'))
            depth_path_list.append(os.path.join(depth_base_path, camID, f'{camID}_{frame}.png'))

        ###################################### HAND ######################################
        ## load mano scale, xyz_root for vert
        anno_path = anno_path_list[0]
        with open(anno_path, 'r', encoding='cp949') as file:
            anno = json.load(file)

        hand_joints = anno['annotations'][0]['data']
        hand_mano_rot = anno['Mesh'][0]['mano_trans']
        hand_mano_pose = anno['Mesh'][0]['mano_pose']
        hand_mano_shape = anno['Mesh'][0]['mano_betas']
        scale = anno['hand']['mano_scale']
        xyz_root = anno['hand']['mano_xyz_root']

        hand_mano_rot = torch.FloatTensor(np.asarray(hand_mano_rot))
        hand_mano_pose = torch.FloatTensor(np.asarray(hand_mano_pose))

        mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1).to(device)
        hand_mano_shape = torch.FloatTensor(np.asarray(hand_mano_shape)).to(device)
        mano_verts, mano_joints = mano_layer(mano_param, hand_mano_shape)

        ## world 3D hand pose for the frame
        mano_verts = (mano_verts / scale) + torch.Tensor(xyz_root).to(device)
        mano_joints = torch.FloatTensor(np.squeeze(np.asarray(anno['annotations'][0]['data']))).to(device)

        ###################################### OBJECT ######################################

        anno_obj_mat = anno['Mesh'][0]['object_mat']
        # for interaction clip
        if len(anno_obj_mat) != 4:
            anno_obj_mat = anno['Mesh'][0]['object_mat'][0]

        obj_mat = torch.FloatTensor(np.asarray(anno_obj_mat)).to(device)

        obj_points = obj_verts_template_h @ obj_mat.T
        # Convert back to Cartesian coordinates
        obj_verts_world = obj_points[:, :3] / obj_points[:, 3:]
        obj_verts_world = obj_verts_world.view(1, -1, 3)

        ###################################### EVALUATION ######################################
        kpts_precision = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
        kpts_recall = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
        kpts_f1 = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}

        mesh_precision = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
        mesh_recall = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
        mesh_f1 = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}

        depth_precision = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
        depth_recall = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}
        depth_f1 = {"mas": 0., "sub1": 0., "sub2": 0., "sub3": 0.}


        for camIdx, anno_path in enumerate(anno_path_list):
            ###################################### VISUALIZATINO ######################################
            camID = cam_list[camIdx]
            Ks = Ks_list[camIdx]
            Ms = Ms_list[camIdx]

            hand_joints = mano_joints
            hand_verts = mano_verts

            ## poses per cam
            joints_cam = torch.unsqueeze(torch.Tensor(mano3DToCam3D(hand_joints, Ms)), axis=0)
            verts_cam = torch.unsqueeze(mano3DToCam3D(hand_verts, Ms), 0)
            verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, Ms), 0)

            ## mesh rendering
            pred_rendered = renderer_set[camIdx].render_meshes([verts_cam, verts_cam_obj],
                                                                    [hand_faces_template, obj_faces_template], flag_rgb=True)

            pred_rendered_hand_only = renderer_set[camIdx].render(verts_cam, hand_faces_template, flag_rgb=True)
            pred_rendered_obj_only = renderer_set[camIdx].render(verts_cam_obj, obj_faces_template, flag_rgb=True)

            rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
            depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
            seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy())
            seg_mesh = np.array(np.ceil(seg_mesh / np.max(seg_mesh)), dtype=np.uint8)

            ## projection on image plane
            pred_kpts2d = projectPoints(joints_cam, Ks)
            pred_kpts2d = np.squeeze(pred_kpts2d.clone().cpu().detach().numpy())
            rgb_2d_pred = paint_kpts(None, rgb_mesh, pred_kpts2d)

            rgb_input = np.asarray(cv2.imread(rgb_path_list[camIdx]))
            depth_input = np.asarray(cv2.imread(depth_path_list[camIdx], cv2.IMREAD_UNCHANGED)).astype(float)

            seg_mask = np.copy(seg_mesh)
            seg_mask[seg_mesh > 0] = 1
            rgb_2d_pred *= seg_mask[..., None]

            ### reproduced visualization result ###
            img_blend_pred = cv2.addWeighted(rgb_input, 1.0, rgb_2d_pred, 0.4, 0)
            cv2.imshow("visualize", img_blend_pred)
            cv2.waitKey(1)

            ## create segment map ##
            hand_depth = np.squeeze(pred_rendered_hand_only['depth'][0].cpu().detach().numpy())
            obj_depth = np.squeeze(pred_rendered_obj_only['depth'][0].cpu().detach().numpy())
            hand_depth[hand_depth == 10] = 0
            hand_depth *= 1000.
            obj_depth[obj_depth == 10] = 0
            obj_depth *= 1000.

            hand_seg_masked = np.where(abs(depth_mesh - obj_depth) < 1.0, 0, 1)
            obj_seg_masked = np.where(abs(depth_mesh - hand_depth) < 1.0, 0, 1)

            cv2.imshow("hand_seg_masked", np.asarray(hand_seg_masked * 255, dtype=np.uint8))
            cv2.imshow("obj_seg_masked", np.asarray(obj_seg_masked * 255, dtype=np.uint8))
            cv2.waitKey(0)


def main():
    rootDir = os.path.join(baseDir, FLAGS.db)
    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(rootDir, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            load_annotation(rootDir, seqName, trialName)



if __name__ == '__main__':
    main()