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

class DataLoader:
    def __init__(self, base_path:str, data_date:str, data_type:str, data_trial:str, cam:str):

        # base_path : os.path.join(os.getcwd(), 'dataset')
        # data_date : 230822
        # data_type : 230822_S01_obj_01_grasp_13
        # data_trial : trial_0
        # cam : 'mas'
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
        self.meta_base_path = os.path.join(self.base_path, 'meta')



        self.cam_path = os.path.join(base_path, data_date+"_cam")

        # #Get data from files
        self.cam_parameter = self.load_cam_parameters()

        self.db_len = len(glob(os.path.join(self.rgb_path, self.cam, '*.jpg')))

        # set preprocessed sample as pickle
        self.sample_dict = {}
        sample_pkl_path = self.base_path + '/sample_'+ cam + '.pickle'
        if os.path.exists(sample_pkl_path):
            print("found pre-loaded pickle of %s" % cam)
            with open(sample_pkl_path, 'rb') as handle:
                self.sample_dict = pickle.load(handle)
        else:
            for frame in tqdm(range(self.db_len)):
                sample = self.load_sample(frame)
                self.sample_dict[frame] = sample
            print("saving pickle of %s" % cam)
            with open(sample_pkl_path, 'wb') as handle:
                pickle.dump(self.sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_sample(self, index):
        if len(self.sample_dict) <= index:
            return None
        else:
            return self.sample_dict[index]

    def load_sample(self, index):
        sample = {}

        # get meta data
        meta = self.get_meta(index)

        if meta is None:
            return None

        bb = np.asarray(meta['bb']).astype(int)
        sample['bb'] = np.copy(bb)
        sample['img2bb'] = meta['img2bb']
        sample['kpts3d'] = meta['kpts']
        sample['kpts2d'] = meta['kpts'][:, :2]

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

        sample['rgb'], sample['depth'], sample['seg'], sample['seg_obj'], _, _ = self.get_img(index)
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

        return [cam_intrinsic, cam_extrinsic, dist_coeff]

    def get_img(self, idx):
        rgb_raw_path = os.path.join(self.rgb_raw_path, self.cam + '_%01d.jpg' % idx)
        depth_raw_path = os.path.join(self.depth_raw_path, self.cam + '_%01d.png' % idx)

        rgb_path = os.path.join(self.rgb_path, self.cam, self.cam+'_%04d.jpg'%idx)
        depth_path = os.path.join(self.depth_path, self.cam, self.cam+'_%04d.png'%idx)

        # currently use temporal segmentation folder
        seg_path = os.path.join(self.seg_path, self.cam, 'raw_seg_results', self.cam+'_%04d.jpg'%idx)
        seg_obj_path = os.path.join(self.seg_path + '_deep', self.cam, 'raw_seg_results', self.cam + '_%04d.jpg' % idx)


        assert os.path.exists(rgb_path)
        assert os.path.exists(depth_path)

        # rgb_raw = np.asarray(cv2.imread(rgb_raw_path))
        # depth_raw = np.asarray(cv2.imread(depth_raw_path, cv2.IMREAD_UNCHANGED)).astype(float)
        rgb_raw = None
        depth_raw = None

        rgb = np.asarray(cv2.imread(rgb_path))
        depth = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)

        # there are skipped frame for segmentation
        if os.path.exists(seg_path):
            seg = np.asarray(cv2.imread(seg_path, cv2.IMREAD_UNCHANGED))
        else:
            seg = np.zeros((CFG_CROP_IMG_HEIGHT, CFG_CROP_IMG_WIDTH))
        seg = np.where(seg>1, 1, 0)

        if os.path.exists(seg_obj_path):
            seg_obj = np.asarray(cv2.imread(seg_obj_path, cv2.IMREAD_UNCHANGED))
            seg_obj[seg_obj == 2] = 0  # bg:0, obj:1, hand:2
        else:
            seg_obj = None

        return rgb, depth, seg, seg_obj, rgb_raw, depth_raw
    
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
        self.obj_pose_dir = os.path.join(self.obj_dir, data_type[:-9])

        obj_type = data_type.split('_')[3]
        obj_class = obj_type + '_' + str(OBJType(int(obj_type)).name)
        self.obj_template_dir = os.path.join(base_path, 'obj_scanned_models', obj_class)

        # load object mesh data (new scanned object need to be load through pytorch3d 'load_obj'
        obj_mesh_path = os.path.join(self.obj_template_dir, obj_class) + '.obj'
        # self.obj_mesh_data = self.read_obj(obj_mesh_path)
        self.obj_mesh_data = {}
        self.obj_mesh_data['verts'], faces, _ = load_obj(obj_mesh_path)
        self.obj_mesh_data['faces'] = faces.verts_idx

        # load from results of preprocess.py
        self.obj_init_pose = None


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
            sample = self.obj_pose_data[str(index)]
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
