import os
from shutil import ExecError
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../', 'utils'))

from modules.utils.loadParameters import LoadCameraMatrix, LoadDistortionParam, LoadCameraMatrix_undistort, LoadCameraParams
import numpy as np
import cv2
import json
import pickle
import torch
import torch.nn as nn
from glob import glob
from config import *
from tqdm import tqdm

from utils.dataUtils import *
import modules.common.transforms as tf

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

'''
데이터셋 구조를 다음과 같이 세팅.(임시)
- base_path
    - DATE (ex. 230612)
        - DATE_TYPE (ex. 230612_bare)
            - depth
            - depth_cam
            - meta
            - rgb
            - rgb_crop
            - segmentation
    - DATE_cam
        - DATE_camerainfo.txt
        - cameraParamsBA.json 
        - mas_intrinsic.json
        - sub1_intrinsic.json
        - sub2_intrinsic.json
        - sub3_intrinsic.json
    - DATE_hand
        - DATA_TYPE (ex. 230612_bare)
        ...
'''

'''
TODO:
    - 데이터 폴더 구조 정리
    - data_type 따로 안넣어줘도 되도록
'''

h = np.array([[0,0,0,1]])

def split_ext(fpath:str):
    '''
    "filepath/filename.ext" -> ("filename", ".ext")
    '''
    return os.path.splitext(os.path.basename(fpath))


def extract_depth(depth_raw, kpts2d, i):
    depth_point = depth_raw[int(kpts2d[i, 1]), int(kpts2d[i, 0])]
    if depth_point == 0.0:
        depth_point = depth_raw[int(kpts2d[i, 1]) + 2, int(kpts2d[i, 0]) + 2]
    if depth_point == 0.0:
        depth_point = depth_raw[int(kpts2d[i, 1]) + 2, int(kpts2d[i, 0]) - 2]
    if depth_point == 0.0:
        depth_point = depth_raw[int(kpts2d[i, 1]) - 2, int(kpts2d[i, 0]) + 2]
    if depth_point == 0.0:
        depth_point = depth_raw[int(kpts2d[i, 1]) - 2, int(kpts2d[i, 0]) - 2]

    return depth_point

class DataLoader:
    def __init__(self, base_path:str, data_date:str, cam_path:str, data_type:str, data_trial:str, cam:str, device='cuda'):

        # base_path : os.path.join(os.getcwd(), 'dataset')
        # data_date : 230822
        # data_type : 230822_S01_obj_01_grasp_13
        # data_trial : trial_0
        # cam : 'mas'
        self.device = device
        self.cam = cam
        self.data_date = data_date
        self.data_type = data_type
        self.data_trial = data_trial
        self.base_path = os.path.join(base_path, data_date, data_type, data_trial)
        self.base_path_result = os.path.join(base_path, data_date + '_result', data_type, data_trial)

        self.rgb_raw_path = os.path.join(self.base_path_result, 'rgb')
        self.depth_raw_path = os.path.join(self.base_path_result, 'depth')

        self.rgb_path = os.path.join(self.base_path, 'rgb_crop')
        self.depth_path = os.path.join(self.base_path, 'depth_crop')
        self.seg_path = os.path.join(self.base_path, 'segmentation')
        self.seg_obj_path = os.path.join(self.base_path, 'segmentation_obj')
        self.meta_base_path = os.path.join(self.base_path, 'meta')

        self.seg_deep_path = os.path.join(self.base_path, 'segmentation_deep')

        self.depth_vis_path = os.path.join(self.base_path, 'depth_vis')

        self.cam_path = os.path.join(base_path, cam_path)

        # #Get data from files
        self.cam_parameter = self.load_cam_parameters()

        self.db_len = len(glob(os.path.join(self.rgb_raw_path, self.cam, '*.jpg')))

        # set preprocessed sample as pickle
        sample_dict = {}
        sample_pkl_path = self.base_path + '/sample_'+ cam + '.pickle'
        if os.path.exists(sample_pkl_path):
            print("found pre-loaded pickle of %s" % cam)
            with open(sample_pkl_path, 'rb') as handle:
                sample_dict = pickle.load(handle)
        else:
            for frame in tqdm(range(self.db_len)):
                sample = self.load_sample(frame)
                sample_dict[frame] = sample
            print("saving pickle of %s" % cam)
            with open(sample_pkl_path, 'wb') as handle:
                pickle.dump(sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.sample_dict, self.sample_kpt = self.sample_to_torch(sample_dict)

    def sample_to_torch(self, sample_dict):
        sample_dict_torch = {}
        sample_kpt = {}
        for idx in range(len(sample_dict)):
            sample = sample_dict[idx]

            if 'bb' not in sample.keys():
                sample_torch = {}
                sample_torch['rgb_raw'] = torch.FloatTensor(sample['rgb_raw']).to(self.device)
                sample_torch['depth_raw'] = torch.unsqueeze(torch.FloatTensor(sample['depth_raw']), 0).to(self.device)

                sample_dict_torch[idx] = sample_torch
                sample_kpt[idx] = None
            else:
                sample_kpt_ = {}
                sample_kpt_['kpts3d'] = np.copy(sample['kpts3d'])

                sample_torch = {}
                sample_torch['bb'] = np.asarray(sample['bb']).astype(int)
                sample_torch['img2bb'] = sample['img2bb']
                sample_torch['kpts2d'] = torch.unsqueeze(torch.FloatTensor(sample['kpts2d']), 0).to(self.device)
                sample_torch['kpts3d'] = torch.unsqueeze(torch.FloatTensor(sample['kpts3d']), 0).to(self.device)
                sample_torch['rgb'] = torch.FloatTensor(sample['rgb']).to(self.device)
                sample_torch['depth'] = torch.unsqueeze(torch.FloatTensor(sample['depth']), 0).to(self.device)
                sample_torch['depth_obj'] = torch.unsqueeze(torch.FloatTensor(sample['depth_obj']), 0).to(self.device)
                sample_torch['seg'] = torch.unsqueeze(torch.FloatTensor(sample['seg']), 0).to(self.device)
                sample_torch['seg_obj'] = torch.unsqueeze(torch.FloatTensor(sample['seg_obj']), 0).to(self.device)

                vis = sample['visibility']
                vis[vis < 1] = CFG_NON_VISIBLE_WEIGHT
                sample_torch['visibility'] = torch.unsqueeze(torch.FloatTensor(vis), 1).to(self.device)

                if CFG_exist_tip_db:
                    if sample['tip2d'] != None:
                        tip2d_np = []
                        tip2d_idx = []
                        for key in sample['tip2d'].keys():
                            tip2d_np.append(sample['tip2d'][key])
                            tip2d_idx.append(CFG_TIP_IDX[key])

                        sample_torch['tip2d'] = torch.unsqueeze(torch.FloatTensor(np.array(tip2d_np)), 0).to(self.device)
                        sample_torch['validtip'] = tip2d_idx
                    else:
                        sample_torch['tip2d'] = None
                        sample_torch['validtip'] = None
                else:
                    sample_torch['tip2d'] = None
                    sample_torch['validtip'] = None

                sample_dict_torch[idx] = sample_torch
                sample_kpt[idx] = sample_kpt_

        return sample_dict_torch, sample_kpt


    def get_sample(self, index):
        if len(self.sample_dict) <= index:
            return None
        else:
            return self.sample_dict[index]

    def load_sample(self, index):
        sample = {}

        # get meta data
        meta = self.get_meta(index)

        if meta['kpts'] is None:
            sample['rgb_raw'], sample['depth_raw'] = self.get_img(index, flag_bb=False)
            return sample
        else:
            bb = np.asarray(meta['bb']).astype(int)
            sample['bb'] = np.copy(bb)
            sample['img2bb'] = meta['img2bb']
            sample['kpts3d'] = meta['kpts']
            sample['kpts2d'] = meta['kpts'][:, :2]

            sample['visibility'] = meta['visibility']
            if '2D_tip_gt' in meta:
                sample['tip2d'] = meta['2D_tip_gt']

            #get imgs
            # sample['rgb'], depth, seg, rgb_raw, depth_raw = self.get_img(index)
            # # masking depth, need to modify
            # depth_bg = depth_raw > 800
            # depth_raw[depth_bg] = 0
            # # currently segmap is often errorneous
            # # depth_raw[seg == 0] = 0
            # sample['depth'] = depth_raw[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            # sample['seg'] = seg[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            # return sample

            sample['rgb'], sample['depth'], sample['depth_obj'], sample['seg'], sample['seg_obj'], \
                rgb_raw, depth_raw = self.get_img(index)

            return sample



    def load_raw_image(self, index):
        _, _, seg, seg_obj, rgb_raw, depth_raw = self.get_img(index)
        return rgb_raw, depth_raw, seg, seg_obj

    def load_cam_parameters(self):
        _, dist_coeffs, extrinsics, _ = LoadCameraParams(os.path.join(self.cam_path, "cameraParams.json"))
        intrinsics = LoadCameraMatrix_undistort(
            os.path.join(self.cam_path, 'cameraInfo_undistort.txt'))

        cam_intrinsic = intrinsics[self.cam]
        dist_coeff = dist_coeffs[self.cam]
        cam_extrinsic = extrinsics[self.cam].reshape(3, 4)
        # scale z axis value as mm to cm
        cam_extrinsic[:, -1] = cam_extrinsic[:, -1] / 10.0

        cam_intrinsic = torch.FloatTensor(cam_intrinsic).to(self.device)
        cam_extrinsic = torch.FloatTensor(cam_extrinsic).to(self.device)

        return [cam_intrinsic, cam_extrinsic, dist_coeff]

    def get_img(self, idx, flag_bb=True):
        rgb_raw_path = os.path.join(self.rgb_raw_path, self.cam, self.cam + '_%01d.jpg' % idx)
        depth_raw_path = os.path.join(self.depth_raw_path, self.cam, self.cam + '_%01d.png' % idx)

        rgb_raw = np.asarray(cv2.imread(rgb_raw_path))
        depth_raw = np.asarray(cv2.imread(depth_raw_path, cv2.IMREAD_UNCHANGED)).astype(float)

        if not flag_bb:
            return rgb_raw, depth_raw
        else:
            rgb_path = os.path.join(self.rgb_path, self.cam, self.cam+'_%04d.jpg'%idx)
            depth_path = os.path.join(self.depth_path, self.cam, self.cam+'_%04d.png'%idx)
            seg_path = os.path.join(self.seg_deep_path, self.cam, 'raw_seg_results',  self.cam + '_%04d.png' % idx)

            assert os.path.exists(rgb_path)
            assert os.path.exists(depth_path)

            rgb = np.asarray(cv2.imread(rgb_path))
            depth = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)

            # depth_vis = depth / np.max(depth)
            # cv2.imshow("rgb", np.asarray(rgb, dtype=np.uint8))
            # cv2.imshow("depth", np.asarray(depth_vis * 255, dtype=np.uint8))
            # cv2.waitKey(1)

            # there are skipped frame for segmentation
            if os.path.exists(seg_path):
                seg = np.asarray(cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)).astype(float)
            else:
                seg = np.zeros((CFG_CROP_IMG_HEIGHT, CFG_CROP_IMG_WIDTH))

            seg_hand = np.where(seg == 1, 1, 0)
            seg_obj = np.where(seg == 2, 1, 0)

            depth_obj = depth.copy()
            depth_obj[seg_obj == 0] = 0
            depth[seg == 0] = 0

            # change depth image to m scale and background value as positive value
            depth /= 1000.
            depth_obj /= 1000.

            depth_obj = np.where(seg != 2, 10, depth)
            depth_hand = np.where(seg != 1, 10, depth)

            # depth_vis_0 = depth_hand / np.max(depth_hand)
            # depth_vis = depth_obj / np.max(depth_obj)
            # cv2.imshow("rgb", np.asarray(rgb, dtype=np.uint8))
            # cv2.imshow("depth_obj", np.asarray(depth_vis * 255, dtype=np.uint8))
            # cv2.imshow("depth_hand", np.asarray(depth_vis_0 * 255, dtype=np.uint8))
            # cv2.imshow("seg", np.asarray(seg *122, dtype=np.uint8))
            # cv2.waitKey(0)

            # seg_hand[seg_hand==0] = 10
            # seg_obj[seg_obj == 0] = 10
            return rgb, depth_hand, depth_obj, seg_hand, seg_obj, rgb_raw, depth_raw
    
    def get_meta(self, idx):
        meta_path = os.path.join(self.meta_base_path, self.cam ,self.cam+'_%04d.pkl'%idx)

        if not os.path.exists(meta_path):
            return None
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        return meta

    def __getitem__(self, index:int):
        try:
            sample = self.get_sample(index)
        except ExecError:
            raise "Error at {}".format(index)
        return sample
    
    def __len__(self):
        return self.db_len


class ObjectLoader:
    def __init__(self, base_path:str, data_date:str, data_type:str, data_trial:str, mas_param):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # base_path : os.path.join(os.getcwd(), 'dataset')
        # data_date : 230822
        # data_type : 230822_S01_obj_01_grasp_13
        # data_trial : trial_0
        self.mas_K, self.mas_M, self.mas_D = mas_param

        self.base_path = base_path
        self.obj_dir = os.path.join(base_path, data_date + '_obj')

        obj_dir_name = "_".join(data_type.split('_')[:-2]) # 230612_S01_obj_01
        self.obj_pose_dir = os.path.join(self.obj_dir, obj_dir_name)

        grasp_idx = data_type.split('_')[-1]
        obj_idx = data_type.split('_')[3]
        obj_name = str(OBJType(int(obj_idx)).name)
        self.obj_class = obj_idx + '_' + obj_name
        self.obj_template_dir = os.path.join(base_path, 'obj_scanned_models', self.obj_class)

        # load object mesh data (new scanned object need to be load through pytorch3d 'load_obj'
        self.obj_mesh_name = self.obj_class + '.obj'
        obj_mesh_path = os.path.join(self.obj_template_dir, self.obj_mesh_name)

        obj_data_name = obj_dir_name + '_grasp_' + str("%02d" % int(grasp_idx)) + '_' + str("%02d" % int(data_trial[-1]))

        obj_scale_data_namme = obj_data_name + '_obj_scale.pkl'
        obj_scale_data_path = os.path.join(self.obj_pose_dir, obj_scale_data_namme)
        with open(obj_scale_data_path, 'rb') as f:
            self.obj_scale = pickle.load(f)

        self.obj_mesh_data = {}
        verts, faces, _ = load_obj(obj_mesh_path)
        if self.obj_class in CFG_OBJECT_SCALE:
             verts *= float(self.obj_scale)

        if self.obj_class in CFG_OBJECT_SCALE_SPECIFIC.keys():
            verts *= float(CFG_OBJECT_SCALE_SPECIFIC[self.obj_class])

        self.obj_mesh_data['verts'] = verts
        self.obj_mesh_data['faces'] = faces.verts_idx

        # load from results of preprocess.py
        obj_pose_data_name = obj_data_name + '_obj_pose.pkl'
        obj_pose_data_path = os.path.join(self.obj_pose_dir, obj_pose_data_name)
        with open(obj_pose_data_path, 'rb') as f:
            self.obj_init_pose = pickle.load(f)


        marker_cam_data_name = obj_data_name + '_marker_cam.pkl'
        marker_cam_data_path = os.path.join(self.obj_pose_dir, marker_cam_data_name)
        with open(marker_cam_data_path, 'rb') as f:
            marker_cam_pose = pickle.load(f)

        self.marker_cam_pose = {}
        for key in marker_cam_pose:
            if marker_cam_pose[key] is None:
                self.marker_cam_pose[key] = None
            else:
                self.marker_cam_pose[key] = torch.FloatTensor(marker_cam_pose[key]).to(self.device)

    def read_obj(self, file_path):
        verts = []
        faces = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):
                    # Parse vertex coordinates
                    vertex = line[2:].split()
                    vertex = [float(coord) for coord in vertex]
                    verts.append(vertex)
                elif line.startswith('f '):
                    # Parse face indices
                    face = line[2:].split()
                    face = [int(index.split('/')[0]) - 1 for index in face]
                    faces.append(face)

        obj_data = {'verts': verts, 'faces': faces}
        return obj_data

    def __getitem__(self, index: int):
        try:
            sample = self.obj_init_pose[str(index)]
            sample = np.asarray(sample)
        except ExecError:
            raise "Error at load object index {}".format(index)
        return sample

    def __len__(self):
        return self.obj_init_pose.shape[0]




if __name__ == "__main__":
    mas_dataloader = DataLoader("/home/workplace/HOnnotate_OXR/dataset", "230612", "bare", "mas")
    sample = mas_dataloader[0]
    print(sample.keys())
    print(sample['bb'])
