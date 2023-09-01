import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import time


def Project(points, cameraParams, intrinsic):
    cameraParams = cameraParams.reshape(-1,3,4)
    points = np.concatenate((points, np.ones((points.shape[0],1))),1).reshape(-1,4,1)
    projection = np.matmul(intrinsic, cameraParams)
    pointsProj = np.matmul(projection, points).squeeze()
    pointsProj /= np.expand_dims(pointsProj[:,-1],1)
    return pointsProj[:,:2]


def Fun(params, intrinsicMatrices, numCameras, numPoints, cameraIndices, pointIndices, imgPoints, weights=None):
    cameraParams = params[:numCameras * 12].reshape((numCameras, 12))
    objPoints = params[numCameras * 12:].reshape((numPoints, 3))
    pointsProj = Project(objPoints[pointIndices], cameraParams[cameraIndices], intrinsicMatrices[cameraIndices])

    reprojection_error = (pointsProj - imgPoints) ** 2
    return np.sum(reprojection_error,1)


def BundleAdjustmentSparsity(numCameras, numPoints, cameraIndices, pointIndices, mas_idx):
    m = cameraIndices.size
    n = numCameras * 12 + numPoints * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(cameraIndices.size)
    for s in range(12):
        A[i[cameraIndices!=mas_idx], cameraIndices[cameraIndices!=mas_idx] * 12 + s] = 1
    
    for s in range(3):
        A[i, numCameras * 12 + pointIndices * 3 + s] = 1
    
    return A


def BundleAdjustment(cameras, objPointsList, imgPointsLeft, imgPointsRight, cameraParams, intrinsicMatrices):
    mas_idx = cameras.index('mas')
    objPoints = np.concatenate(objPointsList, axis=0)
    numCameras = len(cameras)
    numPoints = objPoints.shape[0]
    imgPoints = np.concatenate(imgPointsLeft+imgPointsRight, axis=0)

    # with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
    cameraIndicesLeft = [i for i,x in enumerate(objPointsList) for _ in range(x.shape[0])]
    cameraIndicesRight = [i+1 for i,x in enumerate(objPointsList) for _ in range(x.shape[0])]
    cameraIndices = np.array(cameraIndicesLeft + cameraIndicesRight, dtype=int)
    # with shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
    pointIndices = np.array([*range(numPoints)] * 2, dtype=int)

    intrinsic = np.empty((numCameras,3,3))
    for i, camera in enumerate(cameras):
        intrinsic[i] = intrinsicMatrices[camera]
    
    x0 = np.hstack((cameraParams.ravel(), objPoints.ravel()))
    f0 = Fun(x0, intrinsic, numCameras, numPoints, cameraIndices, pointIndices, imgPoints)
    reprojectionError = np.sqrt(f0.mean())
    print(f"reprojection error is {reprojectionError}")
    plt.plot(np.sqrt(f0))
    plt.show()

    A = BundleAdjustmentSparsity(numCameras, numPoints, cameraIndices, pointIndices, mas_idx)

    t0 = time.time()
    res = least_squares(Fun, x0, jac_sparsity=A, verbose=1, x_scale='jac', ftol=1e-5, method='trf',
                        args=(intrinsic, numCameras, numPoints, cameraIndices, pointIndices, imgPoints))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    reprojectionError = np.sqrt(res.fun.mean())
    print(f"reprojection error is {reprojectionError}")
    plt.plot(np.sqrt(res.fun))
    plt.show()

    return res