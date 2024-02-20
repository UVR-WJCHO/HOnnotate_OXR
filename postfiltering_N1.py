import os
import sys
import pickle
import warnings
warnings.filterwarnings(action='ignore')

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

from modules.deepLabV3plus.oxr_predict import predict as deepSegPredict
from modules.deepLabV3plus.oxr_predict import load_model as deepSegLoadModel
import pandas as pd
import time
# multiprocessing
import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool




### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_integer('start', 0, 'start idx of sequence(ordered)')   ## name ,default, help
flags.DEFINE_integer('end', 50, 'end idx of sequence(ordered)')   ## name ,default, help
FLAGS(sys.argv)

camIDset = ['mas', 'sub1', 'sub2', 'sub3']

baseDir = os.path.join('/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets')
objModelDir = os.path.join(os.getcwd(), 'obj_scanned_models')
mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
csv_save_path = os.path.join(os.getcwd(), 'csv_output')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                       center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)

hand_faces_template = mano_layer.th_faces.repeat(1, 1, 1)


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


def extractBbox(hand_2d, image_rows=1080, image_cols=1920, bbox_w=640, bbox_h=480):
        # consider fixed size bbox
        x_min_ = min(hand_2d[:, 0])
        x_max_ = max(hand_2d[:, 0])
        y_min_ = min(hand_2d[:, 1])
        y_max_ = max(hand_2d[:, 1])

        x_avg = (x_min_ + x_max_) / 2
        y_avg = (y_min_ + y_max_) / 2

        x_min = max(0, x_avg - (bbox_w / 2))
        y_min = max(0, y_avg - (bbox_h / 2))

        if (x_min + bbox_w) > image_cols:
            x_min = image_cols - bbox_w
        if (y_min + bbox_h) > image_rows:
            y_min = image_rows - bbox_h

        bbox = [x_min, y_min, bbox_w, bbox_h]
        return bbox, [x_min_, x_max_, y_min_, y_max_]


class deeplab_opts():
    def __init__(self, object_id):
        self.model = 'deeplabv3plus_mobilenet'
        self.output_stride = 16
        self.gpu_id = '0'
        self.ckpt = "./modules/deepLabV3plus/checkpoints/%02d_best_deeplabv3plus_mobilenet_oxr_os16.pth" % int(object_id)

        assert os.path.isfile(self.ckpt), "no ckpt files for object %02d" % int(object_id)
        # print("...loading ", self.ckpt)
        self.checkpoint = torch.load(self.ckpt, map_location=device)


def load_annotation(base_anno, base_source, seqIdx, seq, seq_count, valid_cam_dict, tqdm_func, global_tqdm):

    with tqdm_func(total=seq_count) as progress:

        ratio_list = []
        seqDir = os.path.join(base_anno, seq)
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):

            valid_cam = valid_cam_dict[trialName]

            progress.set_description(f"{seqIdx} - {seq} - {trialName}")

            #seq ='230822_S01_obj_01_grasp_13'
            # db = seq.split('_')[0]
            # subject_id = seq.split('_')[1][1:]
            obj_id = seq.split('_')[3]
            # grasp_id = seq.split('_')[5]
            # trial_num = trialName.split('_')[1]

            anno_base_path = os.path.join(base_anno, seq, trialName, 'annotation')
            rgb_base_path = os.path.join(base_source, seq, trialName, 'rgb')
            # depth_base_path = os.path.join(base_source, seq, trialName, 'depth')

            ## load each camera parameters ##
            Ks_list = []
            Ms_list = []
            for camID in camIDset:
                if camID in valid_cam:
                    anno_list = os.listdir(os.path.join(anno_base_path, camID))
                    anno_path = os.path.join(anno_base_path, camID, anno_list[0])
                    with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                        anno = json.load(file)

                    Ks = torch.FloatTensor(np.squeeze(np.asarray(anno['calibration']['intrinsic']))).to(device)
                    Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
                    Ms = np.reshape(Ms, (3, 4))
                    ## will be processed in postprocess, didn't.
                    Ms[:, -1] = Ms[:, -1] / 10.0
                    Ms = torch.Tensor(Ms).to(device)

                    Ks_list.append(Ks)
                    Ms_list.append(Ms)
                else:
                    Ks_list.append(None)
                    Ms_list.append(None)

            default_M = np.eye(4)[:3]

            renderer_set = []
            for camIdx, camID in enumerate(camIDset):
                if camID in valid_cam:
                    renderer = Renderer('cuda', 1, default_M, Ks_list[camIdx], (1080, 1920))
                    renderer_set.append(renderer)
                else:
                    renderer_set.append(None)


            ## load hand & object template mesh ##
            target_mesh_class = str(obj_id) + '_' + str(OBJType(int(obj_id)).name)
            obj_mesh_path = os.path.join(baseDir, objModelDir, target_mesh_class, target_mesh_class + '.obj')
            obj_scale = CFG_OBJECT_SCALE_FIXED[int(obj_id)-1]
            obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
            obj_verts_template = (obj_verts * float(obj_scale)).to(device)
            obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(device)

            h = torch.ones((obj_verts_template.shape[0], 1), device=device)
            obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

            ## load deeplab model ##
            opts = deeplab_opts(int(obj_id))
            model, transform, decode_fn = deepSegLoadModel(opts)
            model = model.eval()

            marginSet = [20, 25, 15, 20]


            ## per frame process ##
            for camIdx, camID in enumerate(camIDset):
                if camID in valid_cam:
                    anno_list = natsorted(os.listdir(os.path.join(anno_base_path, camID)))
                    rgb_list = natsorted(os.listdir(os.path.join(rgb_base_path, camID)))
                    # depth_list = os.listdir(os.path.join(depth_base_path, camID))

                    for i in range(len(anno_list)):
                        anno_path = os.path.join(anno_base_path, camID, anno_list[i])
                        rgb_path = os.path.join(rgb_base_path, camID, rgb_list[i])

                        with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
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

                        hand_joints = np.squeeze(np.asarray(hand_joints))
                        xyz_root = hand_joints[0, :]
                        hand_joints_norm = hand_joints - xyz_root
                        dist_anno = hand_joints_norm[1, :] - hand_joints_norm[0, :]

                        mano_joints = np.squeeze(mano_joints.detach().cpu().numpy())
                        dist_mano = mano_joints[1, :] - mano_joints[0, :]
                        scale = np.average(dist_mano / dist_anno)

                        ## world 3D hand pose for the frame
                        mano_verts = (mano_verts / scale) + torch.Tensor(xyz_root).to(device)
                        mano_joints = torch.FloatTensor(np.squeeze(np.asarray(anno['annotations'][0]['data']))).to(device)

                        ###################################### OBJECT ######################################
                        obj_mat = torch.FloatTensor(np.asarray(anno['Mesh'][0]['object_mat'])).to(device)
                        obj_points = obj_verts_template_h @ obj_mat.T
                        obj_verts_world = obj_points[:, :3] / obj_points[:, 3:]
                        obj_verts_world = obj_verts_world.view(1, -1, 3)

                        ###################################### VISUALIZATION ######################################
                        # Ks = Ks_list[camIdx]
                        Ms = Ms_list[camIdx]

                        # hand_joints = mano_joints
                        hand_verts = mano_verts

                        ## poses per cam
                        # joints_cam = torch.unsqueeze(torch.Tensor(mano3DToCam3D(hand_joints, Ms)), axis=0)
                        verts_cam = torch.unsqueeze(mano3DToCam3D(hand_verts, Ms), 0)
                        verts_cam_obj = torch.unsqueeze(mano3DToCam3D(obj_verts_world, Ms), 0)

                        ## mesh rendering
                        # pred_rendered = renderer_set[camIdx].render_meshes([verts_cam, verts_cam_obj],
                        #                                                         [hand_faces_template, obj_faces_template], flag_rgb=True)

                        pred_rendered_hand_only = renderer_set[camIdx].render(verts_cam, hand_faces_template, flag_rgb=True)
                        pred_rendered_obj_only = renderer_set[camIdx].render(verts_cam_obj, obj_faces_template, flag_rgb=True)

                        # rgb_mesh = np.squeeze((pred_rendered['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                        # # depth_mesh = np.squeeze(pred_rendered['depth'][0].cpu().detach().numpy())
                        # seg_mesh = np.squeeze(pred_rendered['seg'][0].cpu().detach().numpy())
                        # seg_mesh = np.array(np.ceil(seg_mesh / np.max(seg_mesh)), dtype=np.uint8)

                        # rgb_hand = np.squeeze((pred_rendered_hand_only['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                        depth_hand = np.squeeze(pred_rendered_hand_only['depth'][0].cpu().detach().numpy())
                        seg_hand = np.squeeze(pred_rendered_hand_only['seg'][0].cpu().detach().numpy())
                        seg_hand = np.array(np.ceil(seg_hand / np.max(seg_hand)), dtype=np.uint8)

                        # rgb_obj = np.squeeze((pred_rendered_obj_only['rgb'][0].cpu().detach().numpy() * 255.0)).astype(np.uint8)
                        depth_obj = np.squeeze(pred_rendered_obj_only['depth'][0].cpu().detach().numpy())
                        # seg_obj = np.squeeze(pred_rendered_obj_only['seg'][0].cpu().detach().numpy())
                        # seg_obj = np.array(np.ceil(seg_obj / np.max(seg_obj)), dtype=np.uint8)

                        seg_hand_only = np.copy(seg_hand)
                        seg_hand_only[depth_hand > depth_obj] = 0

                        ### reproduced visualization result ###
                        rgb_input = np.asarray(cv2.imread(rgb_path))
                        # cv2.imshow("rgb", rgb_input)
                        # cv2.imshow("seg", seg_hand_only*255)

                        ### deepSegmentation ###
                        hand_2d = np.squeeze(np.asarray(anno['hand']['projected_2D_pose_per_cam']))

                        bbox, bbox_s = extractBbox(hand_2d)
                        rgb_crop = rgb_input[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                        # rgb_hand_crop = rgb_hand[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                        # cv2.imshow("rgb_crop", rgb_crop)
                        # cv2.imshow("rgb_hand_crop", rgb_hand_crop)

                        mask, vis_mask = deepSegPredict(model, transform, rgb_crop, decode_fn, device)
                        vis_mask = np.squeeze(np.asarray(vis_mask))
                        hand_mask = np.asarray(vis_mask[:, :, 0] / 128 * 255, dtype=np.uint8)
                        # cv2.imshow("vis_mask", hand_mask)

                        hand_mesh_crop = seg_hand_only[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] * 255

                        ### set ROI for actual hand ###
                        # wrist_2d = hand_2d[0, :] - bbox[0:2]
                        # midroot_2d = hand_2d[9, :] - bbox[0:2]
                        # dx = midroot_2d[0] - wrist_2d[0]
                        # dy = midroot_2d[1] - wrist_2d[1]
                        # vec_dir = [dx, dy]
                        # norm_dir = vec_dir / np.linalg.norm(vec_dir)
                        # vec_orth = [-dy, dx]
                        # norm_orth = vec_orth / np.linalg.norm(vec_orth)
                        # half_range = 120
                        # p1 = wrist_2d + norm_orth * half_range
                        # p2 = wrist_2d - norm_orth * half_range
                        # p3 = p1 + norm_dir * half_range * 2
                        # p4 = p2 + norm_dir * half_range * 2
                        # pts = np.array([p1, p2, p4, p3], dtype=np.int32)
                        # # for k, kpt in enumerate(pts):
                        # #     row = int(kpt[1])
                        # #     col = int(kpt[0])
                        # #     rgb = (0, 0, 255)
                        # #     r = 4
                        # #     cv2.circle(rgb_crop, (col, row), radius=r, thickness=-1, color=rgb)
                        # # cv2.imshow("seg_crop_new", rgb_crop)
                        # roi_mask = np.zeros(rgb_crop.shape, np.uint8)
                        # roi_mask = cv2.fillPoly(roi_mask, [pts], (255, 255, 255))
                        # roi_mask = roi_mask[:, :, 0]
                        # hand_mask[roi_mask == 0] = 0

                        min_x, max_x, min_y, max_y = bbox_s
                        min_x = int(min_x - bbox[0])
                        max_x = int(max_x - bbox[0])
                        min_y = int(min_y - bbox[1])
                        max_y = int(max_y - bbox[1])

                        margin = marginSet[camIdx]

                        roi_mask = np.zeros(rgb_crop.shape, np.uint8)[:, :, 0]
                        roi_mask[min_y-margin:max_y+margin, min_x-margin:max_x+margin] = 255
                        hand_mask[roi_mask == 0] = 0

                        # cv2.imshow("hand_mask after roi crop", hand_mask)
                        # cv2.imshow("seg_crop", hand_mesh_crop)

                        ### calculate silhouette error ###
                        both_mask = np.zeros(hand_mask.shape, np.uint8)
                        both_mask[hand_mask > 0] = 255
                        both_mask[hand_mesh_crop > 0] = 255

                        # cv2.imshow("both_mask", both_mask)

                        gap_1 = both_mask.copy()
                        gap_1[hand_mask > 0] = 0
                        gap_2 = both_mask.copy()
                        gap_2[hand_mesh_crop > 0] = 0
                        gap = gap_1 + gap_2
                        # cv2.imshow("gap", gap)
                        # cv2.waitKey(0)

                        allmask = np.count_nonzero(both_mask)
                        if allmask != 0:
                            ratio = round(np.count_nonzero(gap) / np.count_nonzero(both_mask), 2)
                        else:
                            ratio = 0

                        # print("img name : ", rgb_list[i])
                        # print("ratio : ", ratio)

                        ratio_list.append([trialName, rgb_list[i][:-4], ratio])

                        progress.update()
                        global_tqdm.update()

            for renderer in renderer_set:
                del renderer
            del model

        average_df = pd.DataFrame(ratio_list, columns=['Trial', 'Seq', 'Value'])
        output_name = seq + '.csv'
        save_path = os.path.join(csv_save_path, output_name)
        average_df.to_csv(save_path, index=False)

    return True


def error_callback(result):
    print("Error!")

def done_callback(result):
    # print("Done. Result: ", result)
    return


def main():
    
    torch.multiprocessing.set_start_method('spawn')

    t1 = time.time()
    process_count = 8
    tasks = []

    base_source = os.path.join(baseDir, '1_Source_data')
    base_anno = os.path.join(baseDir, '2_Labeling_data')

    seq_list = natsorted(os.listdir(base_anno))
    print("total sequence # : ", len(seq_list))

    if FLAGS.start != None and FLAGS.end != None:
        seq_list = seq_list[FLAGS.start:FLAGS.end]
        print(f"execute {FLAGS.start} to {FLAGS.end}")

    total_count = 0
    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(base_anno, seqName)

        seq_count = 0
        
        valid_cam = {}
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            valid_cam[trialName] = []
            for camID in camIDset:
                p = os.path.join(seqDir, trialName, 'annotation', camID)
                if os.path.exists(p):
                    temp = len(os.listdir(p))
                    seq_count += temp
                    total_count += temp
                    valid_cam[trialName].append(camID)

        tasks.append((load_annotation, (base_anno, base_source, seqIdx, seqName,seq_count, valid_cam)))

    pool = TqdmMultiProcessPool(process_count)
    with tqdm.tqdm(total=total_count) as global_tqdm:
        # global_tqdm.set_description(f"{seqName} - total : ")
        pool.map(global_tqdm, tasks, error_callback, done_callback)

    proc_time = round((time.time() - t1) / 60., 2)
    print("total process time : %s min" % (str(proc_time)))


if __name__ == '__main__':
    main()