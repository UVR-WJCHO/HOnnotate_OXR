import numpy as np
from scipy.optimize import least_squares
import time
import cv2

h = np.array([[0,0,0,1]]) # array for homogeneous coordinates


def Project(points, cameraParams, intrinsic):
    cameraParams = cameraParams.reshape(-1,3,4)
    points = np.concatenate((points, np.ones((points.shape[0],1))),1).reshape(-1,4,1)
    projection = np.matmul(intrinsic, cameraParams)
    pointsProj = np.matmul(projection, points).squeeze()
    pointsProj /= np.expand_dims(pointsProj[:,-1],1)
    return pointsProj[:,:2]


def Fun(params, objPoints, intrinsicMatrices, cameraParams, cameraIndices, imgPoints):
    R, _ = cv2.Rodrigues(params[:3])
    T = np.expand_dims(params[3:], 1)
    P = np.concatenate((R, T), axis=1)
    P = np.concatenate((P, h), axis=0)
    objPoints = P @ objPoints
    objPoints = objPoints[:3].T
    pointsProj = Project(objPoints, cameraParams[cameraIndices], intrinsicMatrices[cameraIndices])
    reprojection_error = (pointsProj - imgPoints) ** 2
    return np.sum(reprojection_error,1)


def BundleAdjustment(worldProjection, cameras, objPoints, imgPoints, cameraParams, intrinsicMatrices):
    numCameras = len(cameras)
    numPoints = objPoints.shape[0]
    objPoints = np.concatenate((objPoints.T, np.ones((1, numPoints ))), axis=0)

    cameraIndices = []
    extrinsic = np.empty((numCameras,3,4))
    intrinsic = np.empty((numCameras,3,3))
    for i, camera in enumerate(cameras):
        intrinsic[i] = intrinsicMatrices[camera]
        extrinsic[i] = cameraParams[camera].reshape(3,4)
        cameraIndices += [i] * (numPoints // numCameras)
    cameraIndices = np.array(cameraIndices, dtype=int)


    rvec, _ = cv2.Rodrigues(worldProjection[:,:3])
    tvec = worldProjection[:,3]
    x0 = np.concatenate((rvec.squeeze(), tvec))
    f0 = Fun(x0, objPoints, intrinsic, extrinsic, cameraIndices, imgPoints)
    reprojectionError = np.sqrt(f0.mean())
    print(f"reprojection error is {reprojectionError}")

    t0 = time.time()
    res = least_squares(Fun, x0, verbose=2, x_scale='jac', xtol=1e-6, ftol=1e-4, method='trf',
                        args=(objPoints, intrinsic, extrinsic, cameraIndices, imgPoints))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))


    reprojectionError = np.sqrt(res.fun.mean())
    print(f"reprojection error is {reprojectionError}")

    return res, reprojectionError

