import os
from shutil import ExecError
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../', 'utils'))
from utils.loadParameters import LoadCameraMatrix, LoadDistortionParam, LoadCameraMatrix_undistort
import numpy as np
import cv2
import json
import pickle
import torch
import torch.nn as nn
from glob import glob
from config import *
from tqdm import tqdm


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
        self.base_path = os.path.join(base_path, data_date, data_date+'_'+data_type)
        self.rgb_raw_path = os.path.join(self.base_path, 'rgb')
        self.depth_raw_path = os.path.join(self.base_path, 'depth')

        self.rgb_path = os.path.join(self.base_path, 'rgb_crop')
        self.depth_path = os.path.join(self.base_path, 'depth_crop')

        # currently use temporal path from DH.K grabcut results
        # self.seg_path = os.path.join(self.base_path, 'segmentation')
        self.seg_path = os.path.join(self.base_path, 'seg_temp')

        self.meta_base_path = os.path.join(self.base_path, 'meta')

        self.cam_path = os.path.join(base_path, data_date+"_"+'cam')

        self.hand_path = os.path.join(base_path, data_date+"_"+'hand', data_date+"_"+data_type)

        self.obj_data_path = os.path.join(base_path, data_date+'_'+'obj') + '/'+data_date+'_'+data_type+'.txt'

        # #Get data from files
        self.cam_parameter = self.load_cam_parameters()

        self.db_len = len(glob(os.path.join(self.rgb_path, self.cam, '*.png')))

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
        return self.sample_dict[index]

    def load_sample(self, index):
        sample = {}

        # get meta data
        meta = self.get_meta(index)

        bb = np.asarray(meta['bb']).astype(int)
        sample['bb'] = np.copy(bb)
        sample['img2bb'] = meta['img2bb']
        sample['kpts3d'] = meta['kpts']
        sample['kpts2d'] = meta['kpts'][:, :2]

        #get imgs
        sample['rgb'], depth, seg, rgb_raw, depth_raw = self.get_img(index)

        # masking depth, need to modify
        depth_bg = depth_raw > 800
        depth_raw[depth_bg] = 0
        # currently segmap is often errorneous
        # depth_raw[seg == 0] = 0

        sample['depth'] = depth_raw[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
        sample['seg'] = seg[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]

        return sample

    def load_raw_image(self, index):
        _, _, seg, rgb_raw, depth_raw = self.get_img(index)
        return rgb_raw, depth_raw, seg

    def load_cam_parameters(self):
        with open(os.path.join(self.cam_path, "cameraParamsBA.json")) as json_file:
            camera_extrinsics = json.load(json_file)
            camera_extrinsics = np.array((camera_extrinsics[self.cam])).reshape(3, 4)
            # scale z axis value as mm to cm
            camera_extrinsics[:, -1] = camera_extrinsics[:, -1] / 10.0

        camera_intrinsics = LoadCameraMatrix_undistort(os.path.join(self.cam_path, self.data_date + '_cameraInfo_undistort.txt'))
        camera_intrinsics = camera_intrinsics[self.cam]
        dist_coeffs = LoadDistortionParam(os.path.join(self.cam_path, "%s_intrinsic.json"%self.cam))       

        return [camera_intrinsics, camera_extrinsics, dist_coeffs]

    def get_img(self, idx):
        rgb_raw_path = os.path.join(self.rgb_raw_path, self.cam + '_%01d.png' % idx)
        depth_raw_path = os.path.join(self.depth_raw_path, self.cam + '_%01d.png' % idx)

        rgb_path = os.path.join(self.rgb_path, self.cam, self.cam+'_%04d.png'%idx)
        depth_path = os.path.join(self.depth_path, self.cam, self.cam+'_%04d.png'%idx)

        # currently use temporal segmentation folder
        # seg_path = os.path.join(self.seg_path, self.cam, 'raw_seg_results', self.cam+'_%04d.png'%idx)
        seg_path = os.path.join(self.seg_path, self.cam + '_%01d.png' % idx)

        rgb_raw = np.asarray(cv2.imread(rgb_raw_path))
        depth_raw = np.asarray(cv2.imread(depth_raw_path, cv2.IMREAD_UNCHANGED)).astype(float)

        assert os.path.exists(rgb_path)
        assert os.path.exists(depth_path)

        rgb = np.asarray(cv2.imread(rgb_path))
        depth = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)

        # there are skipped frame for segmentation
        if os.path.exists(seg_path):
            seg = np.asarray(cv2.imread(seg_path, cv2.IMREAD_UNCHANGED))
        else:
            seg = np.zeros((CFG_IMG_HEIGHT, CFG_IMG_WIDTH))
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
        return self.db_len


class ObjectLoader:
    def __init__(self, base_path:str, data_date:str, data_type:str,):
        self.obj_result_dir = os.path.join(base_path, data_date) + '_obj'
        self.obj_template_dir = os.path.join(base_path, 'ObjTemplate')

        # load object data
        seq_name = data_date + '_' + data_type

        obj_mesh_path = os.path.join(self.obj_template_dir, data_type) + '.obj'
        self.obj_mesh_data = self.read_obj(obj_mesh_path)

        obj_file_path = os.path.join(self.obj_result_dir, seq_name) + '.txt'
        self.obj_pose_data = self.read_file(obj_file_path)
        # ICG has own main viewpoint
        self.obj_view = self.obj_pose_data[0]

        self.db_len = int((len(self.obj_pose_data) - 1) / 2)

    def read_file(self, file_path):
        lines = []
        with open(file_path, 'r') as file:
            for line in file:
                # Remove newline character at the end of each line
                line = line.strip()
                lines.append(line)
        return lines

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
            sample = self.obj_pose_data[index * 2 + 2].split(',')
            sample = [float(x) for x in sample]
            sample = np.asarray([sample[x:x + 4] for x in range(0, len(sample), 4)])
        except ExecError:
            raise "Error at load object index {}".format(index)
        return sample

    def __len__(self):
        return self.db_len


if __name__ == "__main__":
    mas_dataloader = DataLoader("/home/workplace/HOnnotate_OXR/dataset", "230612", "bare", "mas")
    sample = mas_dataloader[0]
    print(sample.keys())
    print(sample['bb'])