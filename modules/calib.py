import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
import glob
import argparse
import json
import numpy as np
import math
from utils.loadParameters import LoadCameraMatrix, LoadDistortionParam
from utils.bundleAdjustment import BundleAdjustment
from utils.calibration import StereoCalibrate

parser = argparse.ArgumentParser()
parser.add_argument(
    '--db',
    type=str,
    default='230802',
    help='target db Name'
)
parser.add_argument(
    '--cameras',
    type=list,
    default=["mas", "sub3", "sub2", "sub1"],
    help='Sequence Order'
)
parser.add_argument(
    '--base_dir',
    type=str,
    default='/hdd1/donghwan/OXR/HOnnotate_OXR/dataset',
    help='Base directory that contains images with checkerboard'
)
opt = parser.parse_args()

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


class Calibration():
    def __init__(self):

        # stereo calibration will be executed in (0-th, 1-th), (1-th, 2-th), (2-th, 3-th), ...
        self.cameras = opt.cameras
        
        self.imgDirList = []
        for camera_idx in range(len(self.cameras) - 1):
            target_path = os.path.join(opt.base_dir, opt.db, f"{opt.db}_{self.cameras[camera_idx]}_{self.cameras[camera_idx+1]}")
            if os.path.exists(target_path):
                self.imgDirList.append(target_path)
            else:
                target_path = os.path.join(opt.base_dir, opt.db, f"{opt.db}_{self.cameras[camera_idx+1]}_{self.cameras[camera_idx]}")
                self.imgDirList.append(target_path)

        self.resultDir = os.path.join(opt.base_dir, f"{opt.db}_cam")


        assert os.path.exists(os.path.join(self.resultDir, f"{opt.db}_cameraInfo.txt")), 'cameraInfo.txt does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "mas_intrinsic.json")), 'mas_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub1_intrinsic.json")), 'sub1_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub2_intrinsic.json")), 'sub2_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub3_intrinsic.json")), 'sub3_intrinsic.json does not exist'

        self.intrinsic = LoadCameraMatrix(os.path.join(self.resultDir, f"{opt.db}_cameraInfo.txt"))
        self.distCoeffs = {}
        self.distCoeffs["mas"] = LoadDistortionParam(os.path.join(self.resultDir, "mas_intrinsic.json"))
        self.distCoeffs["sub1"] = LoadDistortionParam(os.path.join(self.resultDir, "sub1_intrinsic.json"))
        self.distCoeffs["sub2"] = LoadDistortionParam(os.path.join(self.resultDir, "sub2_intrinsic.json"))
        self.distCoeffs["sub3"] = LoadDistortionParam(os.path.join(self.resultDir, "sub3_intrinsic.json"))

        self.nSize = (6, 5) # the number of checkers
        self.imgInt = 15
        self.minSize = 30
        self.numCameras = len(self.cameras)

        self.imgPointsLeft = []
        self.imgPointsRight = []
        self.objPoints = []

        self.cameraParams = {camera: np.eye(4)[:3] for camera in self.cameras} # initialize extrinsic parameters

    
    def Calibrate(self):
        #TODO: order of cameraParams
        for i in range(self.numCameras - 1):
            left_cam = self.cameras[i]
            right_cam = self.cameras[i+1]
            retval, R, T, pt2dL, pt2dR, pt3dL, pt3dR = StereoCalibrate(self.imgDirList[i], left_cam, right_cam, self.intrinsic, self.distCoeffs,
                    imgInt=self.imgInt, nsize=self.nSize, minSize=self.minSize)
            
            originCameraParams = self.cameraParams[left_cam]
            originCameraParams = np.concatenate((originCameraParams, h), axis=0)
            targetCameraParams = np.concatenate((R,T),axis=1) @ originCameraParams
            self.cameraParams[right_cam] = targetCameraParams

            self.imgPointsLeft.append(pt2dL)
            self.imgPointsRight.append(pt2dR)

            points3d = np.concatenate((pt3dL, np.ones((pt3dL.shape[0],1))), axis=1) # 3d points in i-th camera's coordinate
            params = np.concatenate((self.cameraParams[left_cam],h), axis=0)
            self.objPoints.append((np.linalg.inv(params) @ points3d.T).T[:,:3]) # 3d points in world coordinate (master cameras' coordinate)


    def BA(self):
        cameraParams = np.zeros((self.numCameras, 12)) # flattened extrinsic paramters
        for idx, camera in enumerate(self.cameras):
            cameraParams[idx] = self.cameraParams[camera].ravel()

        result = BundleAdjustment(self.cameras, self.objPoints, self.imgPointsLeft, self.imgPointsRight, cameraParams, self.intrinsic)

        self.cameraParamsBA = result.x[:self.numCameras * 12].reshape(self.numCameras, 12)
        self.objPointsBA = result.x[self.numCameras * 12:]


    def Save(self):
        cameraParamsBA = {camera: self.cameraParamsBA[i].tolist() for i, camera in enumerate(self.cameras)}

        cameraParams = {
            "intrinsic": {camera: self.intrinsic[camera].tolist() for camera in self.intrinsic},
            "dist": {camera: self.distCoeffs[camera].tolist() for camera in self.distCoeffs},
            "extrinsic": cameraParamsBA
        }

        with open(os.path.join(self.resultDir, "cameraParams.json"), "w") as fp:
            json.dump(cameraParams, fp, sort_keys=True, indent=4)


def main():
    calib = Calibration()

    calib.Calibrate()
    calib.BA()
    calib.Save()


if __name__ == '__main__':
    main()