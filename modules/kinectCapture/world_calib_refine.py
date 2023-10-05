import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
import argparse
import numpy as np
import cv2
from utils.loadParameters import LoadCameraParams
from utils.calibration import Resgistration3D
from utils.geometry import uv2xyz
from utils.bundleAdjustmentWorld import BundleAdjustment

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    type=str,
    default='dataset/230905_cam',
    help='target db Name'
)
parser.add_argument(
    '--index',
    type=int,
    default=4,
    help='image index to use'
)
parser.add_argument(
    '--cameras',
    nargs='+',
    default=["mas", "sub1", "sub2", "sub3"],
    help='Sequence Order'
)
parser.add_argument(
    '--world_coordinate',
    nargs='+',
    default=[0, 0, 0, 0],
    help='Sequence Order'
)
opt = parser.parse_args()


base_dir = os.path.join(os.getcwd())


class WorldCalib():
    def __init__(self):

        self.nSize = (6, 5) # the number of checkers
        self.result_dir = os.path.join(base_dir, opt.dir)
        self.intrinsic, self.distCoeffs, self.extrinsics = LoadCameraParams(os.path.join(self.result_dir, "cameraParams.json"))
        image_dir = os.path.join(self.result_dir, "world", "rgb")
        depth_dir = os.path.join(self.result_dir, "world", "depth")

        self.cameras = opt.cameras
        self.world_idx = opt.world_coordinate

        assert len(self.cameras) == len(self.world_idx), "the number of cameras and world idx should be equal"

        self.image_path = {}
        self.depth_path = {}
        for cam in self.cameras:
            self.image_path[cam] = os.path.join(image_dir, cam, f'{cam}_{opt.index}.jpg')
            self.depth_path[cam] = os.path.join(depth_dir, cam, f'{cam}_{opt.index}.png')

    
    def InitWorldCoordinate(self):
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        # mas - 0, sub2 - 3

        detected_corners = {}
        for cam in self.cameras:
            image = cv2.imread(self.image_path[cam])
            print(self.image_path[cam])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # retval, corners = cv2.findChessboardCorners(gray, self.nSize)
            retval, corners = cv2.findChessboardCornersSB(gray, self.nSize, flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # retval, corners = cv2.findChessboardCornersSB(gray, self.nSize, flags = cv2.CALIB_CB_LARGER + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE)

            assert retval, f"all cameras input should valid checkerboard iamge \nthere is no detection {cam}"

            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            detected_corners[cam] = corners
        
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

        objPoints = []
        imgPoints = []

        for cam, idx in zip(self.cameras, self.world_idx):
            format = checker_formats[int(idx)]
            for transposed in format:
                retval, objp, xy, R, T = self.SolvePnP(detected_corners[cam], reorder_idx=format[transposed], transposed=transposed, cam=cam)
                if not retval: continue
                P = np.concatenate((R, T), axis=1)
                objPoints.append(objp)
                imgPoints.append(xy)

        P = np.concatenate((P, h), axis=0)
        projection = self.extrinsics[cam].reshape(3,4)
        projection = np.concatenate((projection, h), axis=0)
        projection = np.linalg.inv(projection)

        world_projection = projection @ P
        world_projection = world_projection[:3]

        objPoints = np.concatenate(objPoints, axis=0)
        imgPoints = np.concatenate(imgPoints, axis=0)

        result, err = BundleAdjustment(world_projection, self.cameras, objPoints, imgPoints, self.extrinsics, self.intrinsic)

        R, _ = cv2.Rodrigues(result.x[:3])
        T = np.expand_dims(result.x[3:], 1)
        P = np.concatenate((R, T), axis=1)
        np.save(os.path.join(self.result_dir, f"global_world.npy"), P)
        P = np.concatenate((P, h), axis=0)

        # x, y, z basis
        coord = np.zeros((4,3))
        coord[1] = [65,0,0]
        coord[2] = [0,65,0]
        coord[3] = [0,0,65]

        coord_homo = np.concatenate((coord.T, np.ones((1,4))), axis=0) # world coordinate
        world_coord = P @ coord_homo # camera's coordinate
        world_coord = world_coord[:3].T

        cameras = ['mas', 'sub1', 'sub2', 'sub3']
        for cam in cameras:
            image = cv2.imread(os.path.join(self.result_dir, "world", "rgb", cam, f"{cam}_{opt.index}.jpg"))
            projection = self.extrinsics[cam].reshape(3,4)
            reprojected, _ = cv2.projectPoints(world_coord, projection[:,:3], projection[:,3:], self.intrinsic[cam], self.distCoeffs[cam])

            O = reprojected[0,0]
            X = reprojected[1,0]
            Y = reprojected[2,0]
            Z = reprojected[3,0]

            xyz = [X,Y,Z]

            for color, basis in zip(colors, xyz):
                image = cv2.line(image, (int(O[0]), int(O[1])), (int(basis[0]), int(basis[1])), color, 2)

            cv2.imwrite(os.path.join(self.result_dir, f"{cam}_world.png"), image)


    def SolvePnP(self, pts_2d, reorder_idx, transposed, cam):
        if transposed:
            objp = np.zeros((self.nSize[0]*self.nSize[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nSize[1],0:self.nSize[0]].T.reshape(-1,2)
        else:
            objp = np.zeros((self.nSize[1]*self.nSize[0],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nSize[0],0:self.nSize[1]].T.reshape(-1,2)
        objp *= 65

        corners = pts_2d[reorder_idx]

        normalized_points = cv2.undistortPoints(corners, self.intrinsic[cam], self.distCoeffs[cam])
        normalized_points = normalized_points.squeeze().transpose()
        normalized_points = np.concatenate((normalized_points, np.ones((1, self.nSize[0] * self.nSize[1]))), 0)
        undistorted_points = np.matmul(self.intrinsic[cam], normalized_points)[:2]
        xy = undistorted_points.transpose()
        depth = cv2.imread(self.depth_path[cam], -1)
        valz, xyz = uv2xyz(self.intrinsic[cam], depth, corners.squeeze(), xy)

        retval, R, T = Resgistration3D(objp[valz], xyz[valz])

        # x, y, z basis
        coord = np.zeros((4,3))
        coord[1] = [1,0,0]
        coord[2] = [0,1,0]
        coord[3] = [0,0,1]
        coord *= 65

        reprojected, _ = cv2.projectPoints(coord, R, T, self.intrinsic[cam], self.distCoeffs[cam])

        O = reprojected[0,0]
        X = reprojected[1,0]
        Y = reprojected[2,0]
        Z = reprojected[3,0]
        xyz = [X, Y, Z]

        retval = (Z[1] - O[1] <= 0)

        return retval, objp, xy, R, T


def main():
    calib = WorldCalib()

    calib.InitWorldCoordinate()

if __name__ == '__main__':
    main()