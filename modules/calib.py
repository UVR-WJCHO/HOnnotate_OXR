import os
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
    '--seq1',
    type=str,
    default='230802_mas_sub3',
    help='Sequence Name for Master and Sub1'
)
parser.add_argument(
    '--seq2',
    type=str,
    default='230802_sub2_sub3',
    help='Sequence Name for Sub1 and Sub2'
)
parser.add_argument(
    '--seq3',
    type=str,
    default='230802_sub1_sub2',
    help='Sequence Name for Sub2 and Sub3'
)
parser.add_argument(
    '--res',
    type=str,
    default='results',
    help='Directory to save result'
)
opt = parser.parse_args()

baseDir = '/hdd1/donghwan/OXR/HOnnotate_OXR/dataset'
h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


class Calibration():
    def __init__(self):
        self.imgDirList = []
        self.imgDirList.append(os.path.join(baseDir, opt.db, opt.seq1))
        self.imgDirList.append(os.path.join(baseDir, opt.db, opt.seq2))
        self.imgDirList.append(os.path.join(baseDir, opt.db, opt.seq3))

        self.resultDir = os.path.join(baseDir, opt.db, opt.res)

        # stereo calibration will be executed in (0-th, 1-th), (1-th, 2-th), (2-th, 3-th), ...
        # self.cameras = ["mas", "sub1", "sub2", "sub3"]
        self.cameras = ["mas", "sub3", "sub2", "sub1"]

        assert os.path.exists(os.path.join(self.resultDir, "230802_cameraInfo.txt")), 'cameraInfo.txt does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "mas_intrinsic.json")), 'mas_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub1_intrinsic.json")), 'sub1_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub2_intrinsic.json")), 'sub2_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub3_intrinsic.json")), 'sub3_intrinsic.json does not exist'

        self.intrinsic = LoadCameraMatrix(os.path.join(self.resultDir, "230802_cameraInfo.txt"))
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

        self.cameraParams = np.zeros((self.numCameras, 12)) # flattened extrinsic paramters
        self.cameraParams[0] = np.eye(4)[:3].ravel() # master camera's coordinate is equal to world coordinate.

    
    def Calibrate(self):
        for i in range(self.numCameras - 1):
            retval, R, T, pt2dL, pt2dR, pt3dL, pt3dR = StereoCalibrate(self.imgDirList[i], self.cameras[i], self.cameras[i+1], self.intrinsic, self.distCoeffs,
                    imgInt=self.imgInt, nsize=self.nSize, minSize=self.minSize)
            originCameraParams = self.cameraParams[i].reshape(3,4)
            originCameraParams = np.concatenate((originCameraParams, h), axis=0)
            targetCameraParams = np.concatenate((R,T),axis=1) @ originCameraParams
            self.cameraParams[i+1] = targetCameraParams.ravel()

            self.imgPointsLeft.append(pt2dL)
            self.imgPointsRight.append(pt2dR)

            points3d = np.concatenate((pt3dL, np.ones((pt3dL.shape[0],1))), axis=1) # 3d points in i-th camera's coordinate
            params = np.concatenate((self.cameraParams[i].reshape((3,4)),h), axis=0)
            self.objPoints.append((np.linalg.inv(params) @ points3d.T).T[:,:3]) # 3d points in world coordinate (master cameras' coordinate)


    def BA(self):
        result = BundleAdjustment(self.cameras, self.objPoints, self.imgPointsLeft, self.imgPointsRight, self.cameraParams, self.intrinsic)

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