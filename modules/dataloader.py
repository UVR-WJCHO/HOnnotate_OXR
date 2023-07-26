import os
from shutil import ExecError
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../', 'utils'))
from utils.loadParameters import LoadCameraMatrix, LoadDistortionParam
import numpy as np
import cv2
import json
import pickle
import torch
import torch.nn as nn
from glob import glob

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

def split_ext(fpath:str):
    '''
    "filepath/filename.ext" -> ("filename", ".ext")
    '''
    return os.path.splitext(os.path.basename(fpath))

class DataLoader:
    def __init__(self, base_path:str, data_date:str, data_type:str, cam:str):
        self.cam = cam
        self.data_date = data_date
        self.data_type = data_type
        self.img_base_path = os.path.join(base_path, data_date, data_date+'_'+data_type)
        self.rgb_raw_path = os.path.join(self.img_base_path, 'rgb')
        self.depth_raw_path = os.path.join(self.img_base_path, 'depth')

        self.rgb_path = os.path.join(self.img_base_path, 'rgb_crop')
        self.depth_path = os.path.join(self.img_base_path, 'depth_crop')
        self.seg_path = os.path.join(self.img_base_path, 'segmentation')
        self.meta_base_path = os.path.join(self.img_base_path, 'meta')

        self.cam_path = os.path.join(base_path, data_date+"_"+'cam')

        self.hand_path = os.path.join(base_path, data_date+"_"+'hand', data_date+"_"+data_type)

        self.obj_data_path = os.path.join(base_path, data_date+'_'+'obj') + '/'+data_date+'_'+data_type+'.txt'

        # #Get data from files
        self.cam_parameter = self.load_cam_parameters()

    def get_sample(self, index):
        sample = {}

        #get imgs
        sample['rgb'], depth, sample['seg'], rgb_raw, depth_raw = self.get_img(index)

        depth_bg = depth > 700
        depth[depth_bg] = 0
        depth[depth == 0] = 0
        sample['depth'] = depth

        #get meta data
        meta = self.get_meta(index)

        sample['bb'] = meta['bb']
        sample['img2bb'] = meta['img2bb']

        sample['kpts3d'] = meta['kpts']
        sample['kpts2d'] = meta['kpts'][:, :2]

        # check img
        # rgb_crop = np.copy(rgb_raw)
        # bb = np.asarray(meta['bb']).astype(int)
        # rgb_crop = rgb_crop[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]
        # cv2.imshow("crop", rgb_crop)
        # cv2.waitKey(0)

        return sample

    def load_cam_parameters(self):
        with open(os.path.join(self.cam_path, "cameraParamsBA.json")) as json_file:
            camera_extrinsics = json.load(json_file)
            camera_extrinsics = np.array((camera_extrinsics[self.cam])).reshape(3, 4)
            # scale z axis value as mm to cm
            camera_extrinsics[:, -1] = camera_extrinsics[:, -1] / 10.0

        camera_intrinsics = LoadCameraMatrix(os.path.join(self.cam_path, self.data_date + '_cameraInfo.txt'))
        camera_intrinsics = camera_intrinsics[self.cam]
        # self.dist_coeffs = {}
        # self.dist_coeffs["mas"] = LoadDistortionParam(os.path.join(self.cam_path, "mas_intrinsic.json"))
        # self.dist_coeffs["sub1"] = LoadDistortionParam(os.path.join(self.cam_path, "sub1_intrinsic.json"))
        # self.dist_coeffs["sub2"] = LoadDistortionParam(os.path.join(self.cam_path, "sub2_intrinsic.json"))
        # self.dist_coeffs["sub3"] = LoadDistortionParam(os.path.join(self.cam_path, "sub3_intrinsic.json"))  
        dist_coeffs = LoadDistortionParam(os.path.join(self.cam_path, "%s_intrinsic.json"%self.cam))       

        return [camera_intrinsics, camera_extrinsics, dist_coeffs]

    def get_img(self, idx):
        rgb_raw_path = os.path.join(self.rgb_raw_path, self.cam + '_%01d.png' % idx)
        depth_raw_path = os.path.join(self.depth_raw_path, self.cam + '_%01d.png' % idx)

        rgb_path = os.path.join(self.rgb_path, self.cam, self.cam+'_%04d.png'%idx)
        depth_path = os.path.join(self.depth_path, self.cam, self.cam+'_%04d.png'%idx)
        seg_path = os.path.join(self.seg_path, self.cam, 'raw_seg_results', self.cam+'_%04d.png'%idx)

        rgb_raw = np.asarray(cv2.imread(rgb_raw_path))
        depth_raw = np.asarray(cv2.imread(depth_raw_path, cv2.IMREAD_UNCHANGED)).astype(float)

        rgb = np.asarray(cv2.imread(rgb_path))
        depth = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)
        seg = np.asarray(cv2.imread(seg_path, cv2.IMREAD_UNCHANGED))
        seg = np.where(seg>1, 1, 0)

        return rgb, depth, seg, rgb_raw, depth_raw
    
    def get_meta(self, idx):
        meta_path = os.path.join(self.meta_base_path, self.cam ,self.cam+'_%04d.pkl'%idx)
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
        return len(glob(os.path.join(self.rgb_path, self.cam, '*.png')))

if __name__ == "__main__":
    mas_dataloader = DataLoader("/home/workplace/HOnnotate_OXR/dataset", "230612", "bare", "mas")
    sample = mas_dataloader[0]
    print(sample.keys())
    print(sample['bb'])