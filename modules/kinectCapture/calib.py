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
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    type=str,
    default='dataset/230822_cam',
    help='target db Name'
)
parser.add_argument(
    '--cameras',
    nargs='+',
    default=["mas", "sub1", "sub2", "sub3"],
    help='Sequence Order'
)
parser.add_argument(
    '--num',
    type=int,
    default=100,
    help='Max frame number used in AzureKinect_calib.py'
)
parser.add_argument("--vis", action="store_true")
opt = parser.parse_args()

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


base_dir = os.path.join(os.getcwd())

class Calibration():
    def __init__(self):

        # stereo calibration will be executed in (0-th, 1-th), (1-th, 2-th), (2-th, 3-th), ...
        self.cameras = opt.cameras
        
        self.imgDirList = []
        for camera_idx in range(len(self.cameras) - 1):
            target_path1 = os.path.join(base_dir, opt.dir, f"{self.cameras[camera_idx]}_{self.cameras[camera_idx+1]}")
            target_path2 = os.path.join(base_dir, opt.dir, f"{self.cameras[camera_idx + 1]}_{self.cameras[camera_idx]}")
            if os.path.exists(target_path1):
                self.imgDirList.append(target_path1)
            elif os.path.exists(target_path2):
                self.imgDirList.append(target_path2)
            else:
                print("wrong directory for calibration")
                exit()

        self.resultDir = os.path.join(base_dir, f"{opt.dir}")

        assert os.path.exists(os.path.join(self.resultDir, "cameraInfo.txt")), 'cameraInfo.txt does not exist, run AzureKinectInfo.py first!'
        assert os.path.exists(os.path.join(self.resultDir, "mas_camInfo.json")), 'mas_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub1_camInfo.json")), 'sub1_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub2_camInfo.json")), 'sub2_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub3_camInfo.json")), 'sub3_intrinsic.json does not exist'

        self.intrinsic = LoadCameraMatrix(os.path.join(self.resultDir, "cameraInfo.txt"))
        self.distCoeffs = {}
        self.distCoeffs["mas"] = LoadDistortionParam(os.path.join(self.resultDir, "mas_camInfo.json"))
        self.distCoeffs["sub1"] = LoadDistortionParam(os.path.join(self.resultDir, "sub1_camInfo.json"))
        self.distCoeffs["sub2"] = LoadDistortionParam(os.path.join(self.resultDir, "sub2_camInfo.json"))
        self.distCoeffs["sub3"] = LoadDistortionParam(os.path.join(self.resultDir, "sub3_camInfo.json"))

        self.nSize = (6, 5) # the number of checkers
        self.imgInt = 5
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
                    imgInt=self.imgInt, numImg=opt.num, nsize=self.nSize, minSize=self.minSize, vis_mode=opt.vis)
            
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