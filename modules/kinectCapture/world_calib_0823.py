import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
import argparse
import numpy as np
import cv2
from utils.loadParameters import LoadCameraParams
from utils.calibration import Resgistration3D
from utils.geometry import uv2xyz

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    type=str,
    default='dataset/230822_cam',
    help='target db Name'
)
parser.add_argument(
    '--camera',
    type=str,
    default="mas",
    help='main camera for world coordinate'
)
parser.add_argument(
    '--index',
    type=int,
    default=5,
    help='image index to use'
)
parser.add_argument(
    '--world_coordinate',
    type=int,
    help='saved world coordinate (ex: 0)'
)
opt = parser.parse_args()


base_dir = os.path.join(os.getcwd())


class WorldCalib():
    def __init__(self):

        self.nSize = (6, 5) # the number of checkers
        self.result_dir = os.path.join(base_dir, opt.dir)
        self.intrinsic, self.distCoeffs, self.extrinsics = LoadCameraParams(os.path.join(self.result_dir, "cameraParams.json"))
        self.camera = "mas"
        image_dir = os.path.join(self.result_dir, "world", "rgb")
        depth_dir = os.path.join(self.result_dir, "world", "depth")

        self.image_path = os.path.join(image_dir, self.camera, f'{self.camera}_{opt.index}.jpg')
        depth_path = os.path.join(depth_dir, self.camera, f'{self.camera}_{opt.index}.png')
        self.depth = cv2.imread(depth_path, -1)

    
    def InitWorldCoordinate(self):
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        if opt.world_coordinate is None:
            print("sss ", self.image_path)
            image = cv2.imread(self.image_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            retval, corners = cv2.findChessboardCorners(gray, self.nSize)

            if retval:
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            # visualize world coordinates
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
                    O, xyz, R, T = self.SolvePnP(corners, reorder_idx=format[transposed], transposed=transposed, cam=self.camera)

                    Z = xyz[2]
                    if (Z[1] - O[1] > 0):
                        # if z-basis is down direction, skip it.
                        continue

                    for color, basis in zip(colors, xyz):
                        image = cv2.line(image, (int(O[0]), int(O[1])), (int(basis[0]), int(basis[1])), color, 3)

                    cv2.putText(image, f"{i}-th coord", (int(Z[0]), int(Z[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                    P = np.concatenate((R, T), axis=1)
                    np.save(os.path.join(self.result_dir, f"{i}-world.npy"), P)

            cv2.imwrite(os.path.join(self.result_dir, "world_coordinate.png"), image)

        else:
            P = np.load(os.path.join(self.result_dir, f"{opt.world_coordinate}-world.npy"))
            P = np.concatenate((P, h), axis=0)

            # x, y, z basis
            coord = np.zeros((4,3))
            coord[1] = [65,0,0]
            coord[2] = [0,65,0]
            coord[3] = [0,0,65]

            coord_homo = np.concatenate((coord.T, np.ones((1,4))), axis=0) # world coordinate
            world_coord = P @ coord_homo # camera's coordinate
            projection = self.extrinsics[self.camera].reshape(3,4)
            projection = np.concatenate((projection, h), axis=0)
            projection = np.linalg.inv(projection)
            world_coord = projection @ world_coord # master's coordinate
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
        objp[:,:2] -= 1
        objp *= 65

        corners = pts_2d[reorder_idx]

        normalized_points = cv2.undistortPoints(corners, self.intrinsic[cam], self.distCoeffs[cam])
        normalized_points = normalized_points.squeeze().transpose()
        normalized_points = np.concatenate((normalized_points, np.ones((1, self.nSize[0] * self.nSize[1]))), 0)
        undistorted_points = np.matmul(self.intrinsic[cam], normalized_points)[:2]
        xy = undistorted_points.transpose()
        valz, xyz = uv2xyz(self.intrinsic[cam], self.depth, corners.squeeze(), xy)

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

        return O, xyz, R, T


def main():
    calib = WorldCalib()

    calib.InitWorldCoordinate()

if __name__ == '__main__':
    main()