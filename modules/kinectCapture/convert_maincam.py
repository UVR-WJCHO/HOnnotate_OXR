import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
import argparse
import numpy as np
from utils.loadParameters import LoadCameraParams
import json

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    type=str,
    default='dataset/230905_cam',
    help='target db Name'
)

opt = parser.parse_args()


base_dir = os.path.join(os.getcwd())
result_dir = os.path.join(base_dir, opt.dir)
intrinsic, distCoeffs, extrinsics = LoadCameraParams(os.path.join(result_dir, "cameraParams.json"))
cameras = ['mas', 'sub1', 'sub2', 'sub3']

inv_mas = extrinsics['mas'].reshape(3,4)
inv_mas = np.concatenate((inv_mas, h), axis=0)
inv_mas = np.linalg.inv(inv_mas)

new_extrinsics = {}

for cam in cameras:
    new_extrinsics[cam] = extrinsics[cam].reshape(3,4) @ inv_mas

new_extrinsics = {cam: new_extrinsics[cam].flatten().tolist() for cam in cameras}

with open(os.path.join(result_dir, "cameraParams.json"), "r") as file:
    json_data = json.load(file)
    json_data.update({
        "extrinsic": new_extrinsics
    })

with open(os.path.join(result_dir, "cameraParams.json"), "w") as file:
    json.dump(json_data, file, sort_keys=True, indent=4)

print("complete")