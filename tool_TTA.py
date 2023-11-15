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
flags.DEFINE_string('db', 'test_result', 'target db Name')   ## name ,default, help
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

        Ks = np.asarray(str(anno['calibration']['intrinsic']).split(','), dtype=float)
        Ks = torch.FloatTensor([[Ks[0], 0, Ks[2]], [0, Ks[1], Ks[3]], [0, 0, 1]]).to(device)
        Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
        Ms = np.reshape(Ms, (3, 4))
        ## will be processed in postprocess
        # Ms[:, -1] = Ms[:, -1] / 10.0
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

        hand_mano_rot = torch.FloatTensor(np.asarray(hand_mano_rot))
        hand_mano_pose = torch.FloatTensor(np.asarray(hand_mano_pose))

        mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1).to(device)
        hand_mano_shape = torch.FloatTensor(np.asarray(hand_mano_shape)).to(device)
        mano_verts, mano_joints = mano_layer(mano_param, hand_mano_shape)

        mano_joints = np.squeeze(mano_joints.detach().cpu().numpy())
        hand_joints = np.squeeze(np.asarray(hand_joints))
        xyz_root = hand_joints[0, :]
        hand_joints_norm = hand_joints - xyz_root
        dist_anno = hand_joints_norm[1, :] - hand_joints_norm[0, :]
        dist_mano = mano_joints[1, :] - mano_joints[0, :]
        scale = np.average(dist_mano / dist_anno)

        ## world 3D hand pose for the frame
        mano_verts = (mano_verts / scale) + torch.Tensor(xyz_root).to(device)
        mano_joints = torch.FloatTensor(np.squeeze(np.asarray(anno['annotations'][0]['data']))).to(device)

        ###################################### OBJECT ######################################

        obj_mat = torch.FloatTensor(np.asarray(anno['Mesh'][0]['object_mat'])).to(device)
        ## will be processed in postprocess
        # obj_mat[:3, -1] *= 0.1
        obj_points = obj_verts_template_h @ obj_mat.T
        # Convert back to Cartesian coordinates
        obj_verts_world = obj_points[:, :3] / obj_points[:, 3:]
        obj_verts_world = obj_verts_world.view(1, -1, 3)

        # for i in range(4):
        #     # origin
        #     Ms = Ms_list[i]
        #     verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, Ms), 0)
        #
        #     # new
        #     debug = obj_verts_template_h @ obj_mat.T @ Ms.T
        #     print(".")
        #
        #     obj_mat_cam = (obj_mat.T @ Ms.T).T
        #     debug_1 = obj_verts_template_h @ obj_mat_cam.T
        #     print(".")


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
            with open(anno_path, 'r', encoding='cp949') as file:
                anno = json.load(file)
            camID = cam_list[camIdx]
            Ks = Ks_list[camIdx]
            Ms = Ms_list[camIdx]

            hand_joints = mano_joints
            hand_verts = mano_verts

            ## poses per cam
            joints_cam = torch.unsqueeze(torch.Tensor(mano3DToCam3D(hand_joints, Ms)), axis=0)
            verts_cam = torch.unsqueeze(mano3DToCam3D(hand_verts, Ms), 0)
            verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, Ms), 0)
            # (same) verts_cam_obj = torch.unsqueeze(obj_verts_world @ obj_mat_cam.T, 0)

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
            cv2.waitKey(0)

            ###################################### EVALUATION ######################################

            # ### load GT kpt data (human-annotated 2D tip data ###
            # gt_kpts2d = np.zeros((21, 2))
            #
            # ## need seg gt and bbox
            # gt_seg = None
            # gt_seg_obj = None
            # bbox = None
            #
            #
            # hand_depth = np.squeeze(pred_rendered_hand_only['depth'][0].cpu().detach().numpy())
            # obj_depth = np.squeeze(pred_rendered_obj_only['depth'][0].cpu().detach().numpy())
            # hand_depth[hand_depth == 10] = 0
            # hand_depth *= 1000.
            # obj_depth[obj_depth == 10] = 0
            # obj_depth *= 1000.
            #
            # hand_seg_masked = np.where(abs(depth_mesh - obj_depth) < 1.0, 0, 1)
            # obj_seg_masked = np.where(abs(depth_mesh - hand_depth) < 1.0, 0, 1)
            #
            # hand_seg_masked = hand_seg_masked[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            # obj_seg_masked = obj_seg_masked[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            #
            #
            # # 1. 3D keypoints F1-Score
            # TP = 1e-7  # 각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이내
            # FP = 1e-7  # 각 키포인트의 픽셀 좌표가 참값의 픽셀 좌표와 유클리디안 거리 50px 이상
            # FN = 1e-7  # 참값이 존재하지만 키포인트 좌표가 존재하지 않는 경우(미태깅)
            # for idx, gt_kpt in enumerate(gt_kpts2d):
            #     pred_kpt = pred_kpts2d[idx]
            #     if (pred_kpt == None).any():
            #         FN += 1
            #     dist = np.linalg.norm(gt_kpt - pred_kpt)
            #     if dist < 50:
            #         TP += 1
            #     elif dist >= 50:
            #         FP += 1
            #
            # keypoint_precision_score = TP / (TP + FP)
            # keypoint_recall_score = TP / (TP + FN)
            # keypoint_f1_score = 2 * (keypoint_precision_score * keypoint_recall_score /
            #                          (keypoint_precision_score + keypoint_recall_score))  # 2*TP/(2*TP+FP+FN)
            # kpts_precision[camID] = keypoint_precision_score
            # kpts_recall[camID] = keypoint_recall_score
            # kpts_f1[camID] = keypoint_f1_score
            #
            # # 2. mesh pose F1-Score
            # TP = 0  # 렌더링된 이미지의 각 픽셀의 segmentation 클래스(background, object, hand)가 참값(실제 RGB- segmentation map)의 클래스와 일치
            # FP = 0  # 렌더링된 이미지의 각 픽셀의 segmentation 클래스가 참값의 클래스와 불일치
            # FN = 0  # 참값이 존재하지만 키포인트 좌표의 segmentation class가 존재하지 않는 경우(미태깅) ??
            # gt_seg_hand = np.squeeze((gt_seg[0].cpu().detach().numpy()))
            # gt_seg_obj = np.squeeze((gt_seg_obj[0].cpu().detach().numpy()))
            #
            # TP = np.sum(np.where(hand_seg_masked > 0, hand_seg_masked == gt_seg_hand, 0)) + \
            #      np.sum(np.where(obj_seg_masked > 0, obj_seg_masked == gt_seg_obj, 0))
            # FP = np.sum(np.where(hand_seg_masked > 0, hand_seg_masked != gt_seg_hand, 0)) + \
            #      np.sum(np.where(obj_seg_masked > 0, obj_seg_masked != gt_seg_obj, 0))
            # # seg_masked_FN = (gt_seg_hand > 0) * (hand_seg_masked == 0) \
            # #                 + (gt_seg_obj > 0) * (obj_seg_masked == 0)
            #
            # if TP == 0:
            #     TP = 1e-7
            # FN = 1e-7  # np.sum(seg_masked_FN)
            #
            # mesh_seg_precision_score = TP / (TP + FP)
            # mesh_seg_recall_score = TP / (TP + FN)
            # mesh_seg_f1_score = 2 * (mesh_seg_precision_score * mesh_seg_recall_score /
            #                          (mesh_seg_precision_score + mesh_seg_recall_score))  # 2*TP/(2*TP+FP+FN)
            #
            # mesh_precision[camID] = mesh_seg_precision_score
            # mesh_recall[camID] = mesh_seg_recall_score
            # mesh_f1[camID] = mesh_seg_f1_score
            #
            # # 3. hand depth accuracy
            # TP = 1e-7  # 각 키포인트의 렌더링된 깊이값이 참값(실제 깊이영상)의 깊이값과 20mm 이내
            # FP = 1e-7  # 각 키포인트의 렌더링된 깊이값이 참값(실제 깊이영상)의 깊이값과 20mm 이상
            # FN = 1e-7  # 참값이 존재하지만 키포인트 좌표의 깊이값이 존재하지 않는 경우(미태깅)
            #
            # # pred_kpts2d
            # # obj_depth, hand_depth
            # # self.gt_depth, self.gt_depth_obj
            # # gt_depth_hand = np.squeeze(self.gt_depth.cpu().numpy())
            # gt_depth_all = self.gt_depth_raw[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            #
            # all_seg_mask = hand_seg_masked + obj_seg_masked
            # all_seg_mask[all_seg_mask == 2] = 1
            # gt_depth_all *= all_seg_mask
            #
            # # gt_depth_hand[gt_depth_hand==10] = 0
            # # gt_depth_hand *= 1000.
            # # cv2.imshow("gt_depth_all", np.array(gt_depth_all / 100 * 255, dtype=np.uint8))
            # # cv2.imshow("both_depth", np.array(both_depth / 100 * 255, dtype=np.uint8))
            # # cv2.waitKey(0)
            #
            # all_diff = np.abs(gt_depth_all - both_depth)
            #
            # all_FN = np.copy(all_diff)
            # all_FN = all_FN[np.isin(all_diff, gt_depth_all)]
            # all_FN[all_FN > 0] = 1
            # all_FN = all_FN.sum()
            # all_diff[all_diff > 50] = 0  # consider as unlabelled(FN)/noise if error is larger than 5cm
            #
            # # count # of diff > 20
            # all_FP = np.copy(all_diff)
            # all_FP[all_FP <= 20] = 0
            # all_FP[all_FP > 20] = 1
            # all_FP = all_FP.sum()
            #
            # # count only diff < 20 value
            # all_diff[all_diff == 0] = 100
            # all_diff[all_diff <= 20] = 1
            # all_diff[all_diff > 20] = 0
            # all_TP = all_diff.sum()
            #
            # mesh_depth_precision_score = all_TP / (all_TP + all_FP)
            # mesh_depth_recall_score = all_TP / (all_TP + all_FN)
            # mesh_depth_f1_score = 2 * (mesh_depth_precision_score * mesh_depth_recall_score /
            #                            (mesh_depth_precision_score + mesh_depth_recall_score))
            #
            # depth_precision[camID] = mesh_depth_precision_score
            # depth_recall[camID] = mesh_depth_recall_score
            # depth_f1[camID] = mesh_depth_f1_score





def main():
    rootDir = os.path.join(baseDir, FLAGS.db)
    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(rootDir, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            load_annotation(rootDir, seqName, trialName)



if __name__ == '__main__':
    main()