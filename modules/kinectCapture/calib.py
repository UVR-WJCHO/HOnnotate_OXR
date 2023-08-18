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
    default='230817',
    help='target db Name'
)
parser.add_argument(
    '--cameras',
    nargs='+',
    default=["mas", "sub1", "sub2", "sub3"],
    help='Sequence Order'
)
parser.add_argument(
    '--world_img',
    type=str,
    help='Checkerboard image for initializing world coordinate'
)
opt = parser.parse_args()

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


base_dir = os.path.join(os.getcwd())

class Calibration():
    def __init__(self):

        # stereo calibration will be executed in (0-th, 1-th), (1-th, 2-th), (2-th, 3-th), ...
        self.cameras = opt.cameras
        
        self.imgDirList = []
        for camera_idx in range(len(self.cameras) - 1):
            target_path = os.path.join(base_dir, opt.dir, f"{opt.dir}_{self.cameras[camera_idx]}_{self.cameras[camera_idx+1]}")
            if os.path.exists(target_path):
                self.imgDirList.append(target_path)
            else:
                target_path = os.path.join(base_dir, opt.dir, f"{opt.dir}_{self.cameras[camera_idx+1]}_{self.cameras[camera_idx]}")
                self.imgDirList.append(target_path)

        self.resultDir = os.path.join(base_dir, f"{opt.dir}")


        assert os.path.exists(os.path.join(self.resultDir, f"{opt.dir}_cameraInfo.txt")), 'cameraInfo.txt does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "mas_camInfo.json")), 'mas_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub1_camInfo.json")), 'sub1_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub2_camInfo.json")), 'sub2_intrinsic.json does not exist'
        assert os.path.exists(os.path.join(self.resultDir, "sub3_camInfo.json")), 'sub3_intrinsic.json does not exist'

        self.intrinsic = LoadCameraMatrix(os.path.join(self.resultDir, f"{opt.dir}_cameraInfo.txt"))
        self.distCoeffs = {}
        self.distCoeffs["mas"] = LoadDistortionParam(os.path.join(self.resultDir, "mas_camInfo.json"))
        self.distCoeffs["sub1"] = LoadDistortionParam(os.path.join(self.resultDir, "sub1_camInfo.json"))
        self.distCoeffs["sub2"] = LoadDistortionParam(os.path.join(self.resultDir, "sub2_camInfo.json"))
        self.distCoeffs["sub3"] = LoadDistortionParam(os.path.join(self.resultDir, "sub3_camInfo.json"))

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
    

    def InitWorldCoordinate(self, image_path):
        camera = os.path.split(image_path)[-1].split('_')[0]
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(gray, self.nSize)

        if retval:
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # visualize world coordinates
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        checker_formats = [
            {
                0: np.arange(30),
                1: np.arange(30).reshape(5,6).transpose().ravel()
            },
            {
                0: np.arange(30).reshape(5,6)[:,::-1].ravel(),
                1: np.arange(30).reshape(5,6)[:,::-1].transpose().ravel()
            },
            {
                0: np.arange(30).reshape(5,6)[::-1,:].ravel(),
                1: np.arange(30).reshape(5,6)[::-1,:].transpose().ravel()
            },
            {
                0: np.arange(30)[::-1],
                1: np.arange(30).reshape(5,6).transpose().ravel()[::-1]
            },
        ]

        for i, format in enumerate(checker_formats):
            for transposed in format:
                O, xyz, rvec, tvec = self.VisualizeWorldCoordinate(corners, reorder_idx=format[transposed], transposed=transposed, cam=camera)

                Z = xyz[2]
                if (Z[1] - O[1] > 0):
                    # if z-basis is down direction, skip it.
                    continue

                for color, basis in zip(colors, xyz):
                    image = cv2.line(image, (int(O[0]), int(O[1])), (int(basis[0]), int(basis[1])), color, 3)

                cv2.putText(image, f"{i}-th coord", (int(Z[0]), int(Z[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                R = cv2.Rodrigues(rvec)[0]
                P = np.concatenate((R, tvec), axis=1)
                np.save(os.path.join(self.resultDir, f"{i}-world.npy"), P)

        cv2.imwrite(os.path.join(self.resultDir, "world_coordinate.png"), image)


    def VisualizeWorldCoordinate(self, pts_2d, reorder_idx, transposed, cam):
        if transposed:
            objp = np.zeros((self.nSize[0]*self.nSize[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nSize[1],0:self.nSize[0]].T.reshape(-1,2)
        else:
            objp = np.zeros((self.nSize[1]*self.nSize[0],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nSize[0],0:self.nSize[1]].T.reshape(-1,2)

        corners = pts_2d[reorder_idx]
        ret, rvec, tvec = cv2.solvePnP(objp, corners, self.intrinsic[cam], self.distCoeffs[cam])

        # x, y, z basis
        coord = np.zeros((4,3))
        coord[1] = [1,0,0]
        coord[2] = [0,1,0]
        coord[3] = [0,0,1]

        reprojected, _ = cv2.projectPoints(coord, rvec, tvec, self.intrinsic[cam], self.distCoeffs[cam])

        O = reprojected[0,0]
        X = reprojected[1,0]
        Y = reprojected[2,0]
        Z = reprojected[3,0]
        xyz = [X, Y, Z]

        return O, xyz, rvec, tvec


def main():
    calib = Calibration()

    calib.Calibrate()
    calib.BA()
    calib.Save()

    if (opt.world_img != None):
        calib.InitWorldCoordinate(opt.world_img)

if __name__ == '__main__':
    main()