import os
import pickle

import numpy as np
from natsort import natsorted
from modules.utils.loadParameters import LoadCameraMatrix, LoadDistortionParam, LoadCameraMatrix_undistort, LoadCameraParams
from config import *


_, dist_coeffs, extrinsics, _ = LoadCameraParams("D:/HOnnotate_OXR/dataset/231006_cam/cameraParams.json")

ext = extrinsics['mas'].reshape(3, 4)
h = np.expand_dims(np.array([0, 0, 0, 1]), axis=0)
ext_h = np.concatenate((ext, h), axis=0)
main_inv = np.linalg.inv(ext_h)

new_list = []
for i in range(4):
    ext = extrinsics[CFG_VALID_CAM[i]].reshape(3, 4)
    ext_h = np.concatenate((ext, h), axis=0)
    ext_new = ext_h @ main_inv
    new_list.append(ext_new)

print("check new ext")

#
# rootDir = "C:/Projects/OXR_projects/HOnnotate_OXR/dataset/230922_obj"
# seq_list = natsorted(os.listdir(rootDir))
# for seqIdx, seqName in enumerate(seq_list):
#     if seqIdx < 4:
#         continue
#     if seqIdx == 34:
#         break
#     seqDir = os.path.join(rootDir, seqName)
#     files = os.listdir(seqDir)
#     files_scale = [file for file in files if file.endswith("obj_scale.pkl")]
#
#     file_scale = files_scale[0]
#
#     scale_path = os.path.join(seqDir, file_scale)
#     with open(scale_path, 'rb') as f:
#         scale = pickle.load(f)
#
#     print(scale)