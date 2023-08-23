import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import math
from utils.geometry import uv2xyz


def CvCornerFinder(path, nsize):
    color = cv2.imread(path)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray,nsize)

    if retval:
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    '''
    if calibration raises error, uncomment it and check the result of detection
    '''
    #     cv2.drawChessboardCorners(color, nsize, corners, retval)
    # cv2.imshow(path, color)
    # cv2.waitKey(0)

    return retval, corners


def Resgistration3D(p1_t,p2_t):
    #Take transpose as columns should be the points
    p1 = p1_t.transpose()
    p2 = p2_t.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))

    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    # #Calculate rotation matrix
    V = V_t.transpose()
    U_transpose = U.transpose()
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)

    #Check result
    result = T + np.matmul(R,p1)

    retval = np.sqrt(((result - p2) ** 2).mean())
    print("retval", retval)

    return retval, R, T


def StereoCalibrate(imgDir, cam1, cam2, intrinsicMatrices, distCoeffs, imgInt=5, numImg=300, nsize=(12,7), minSize=30):
    rgbDir = os.path.join(imgDir, "rgb")
    depthDir = os.path.join(imgDir, "depth")

    leftIntrinsic = intrinsicMatrices[cam1]
    leftDistCoeffs = distCoeffs[cam1]
    rightIntrinsic = intrinsicMatrices[cam2]
    rightDistCoeffs = distCoeffs[cam2]

    pt2dL = []
    pt2dR = []
    pt3dL =[]
    pt3dR =[]
    nptperImg = nsize[0] * nsize[1]
    numDetection = 0


    print(f"Stereo Calibration between {cam1} and {cam2} starts!")
    for i in tqdm(range(imgInt-1, numImg, imgInt)):
        if not os.path.exists(os.path.join(rgbDir, cam1, f"{cam1}_{i}.jpg")):
            continue
        if not os.path.exists(os.path.join(rgbDir, cam2, f"{cam2}_{i}.jpg")):
            continue

        isValidLeft, leftCorners = CvCornerFinder(os.path.join(rgbDir, cam1, f"{cam1}_{i}.jpg"), nsize)
        isValidRight, rightCorners = CvCornerFinder(os.path.join(rgbDir, cam2, f"{cam2}_{i}.jpg"), nsize)

        if isValidLeft and isValidRight and (
            min(np.linalg.norm(leftCorners[0,0]-leftCorners[1,0]), np.linalg.norm(leftCorners[0,0]-leftCorners[nsize[0],0])) > minSize and
            min(np.linalg.norm(leftCorners[nsize[0]-1,0]-leftCorners[nsize[0]-2,0]), np.linalg.norm(leftCorners[nsize[0]-1,0]-leftCorners[2*nsize[0]-1,0])) > minSize and
            min(np.linalg.norm(leftCorners[-1*nsize[0],0]-leftCorners[-1*nsize[0]+1,0]), np.linalg.norm(leftCorners[-1*nsize[0],0]-leftCorners[-2*nsize[0],0])) > minSize and
            min(np.linalg.norm(leftCorners[-1,0]-leftCorners[-2,0]), np.linalg.norm(leftCorners[-1,0]-leftCorners[-1-nsize[0],0])) > minSize
        ) and (
            min(np.linalg.norm(rightCorners[0,0]-rightCorners[1,0]), np.linalg.norm(rightCorners[0,0]-rightCorners[nsize[0],0])) > minSize and
            min(np.linalg.norm(rightCorners[nsize[0]-1,0]-rightCorners[nsize[0]-2,0]), np.linalg.norm(rightCorners[nsize[0]-1,0]-rightCorners[2*nsize[0]-1,0])) > minSize and
            min(np.linalg.norm(rightCorners[-1*nsize[0],0]-rightCorners[-1*nsize[0]+1,0]), np.linalg.norm(rightCorners[-1*nsize[0],0]-rightCorners[-2*nsize[0],0])) > minSize and
            min(np.linalg.norm(rightCorners[-1,0]-rightCorners[-2,0]), np.linalg.norm(rightCorners[-1,0]-rightCorners[-1-nsize[0],0])) > minSize
        ):
            normalizedPointsLeft = cv2.undistortPoints(leftCorners, leftIntrinsic, leftDistCoeffs)
            normalizedPointsLeft = normalizedPointsLeft.squeeze().transpose()
            normalizedPointsLeft = np.concatenate((normalizedPointsLeft, np.ones((1,nptperImg))),0)
            undistortedPointsLeft = np.matmul(leftIntrinsic, normalizedPointsLeft)[:2,:]
            lxy = undistortedPointsLeft.transpose()
            leftDepth = cv2.imread(os.path.join(depthDir, cam1, f"{cam1}_{i}.png"),-1)
            lvalz, lxyz = uv2xyz(leftIntrinsic, leftDepth, leftCorners.squeeze(), lxy)

            normalizedPointsRight = cv2.undistortPoints(rightCorners, rightIntrinsic, rightDistCoeffs)
            normalizedPointsRight = normalizedPointsRight.squeeze().transpose()
            normalizedPointsRight = np.concatenate((normalizedPointsRight, np.ones((1,nptperImg))),0)
            undistortedPointsRight = np.matmul(rightIntrinsic, normalizedPointsRight)[:2,:]
            rxy = undistortedPointsRight.transpose()
            rightDepth = cv2.imread(os.path.join(depthDir, cam2, f"{cam2}_{i}.png"),-1)
            rvalz, rxyz = uv2xyz(rightIntrinsic, rightDepth, rightCorners.squeeze(), rxy)

            valz = [x and y for x, y in zip(lvalz, rvalz)]

            numDetection += len(valz)
            pt2dL.append(lxy[valz])
            pt2dR.append(rxy[valz])
            pt3dL.append(lxyz[valz])
            pt3dR.append(rxyz[valz])

    pt2dL = np.vstack(pt2dL)
    pt2dR = np.vstack(pt2dR)
    pt3dL = np.vstack(pt3dL)
    pt3dR = np.vstack(pt3dR)

    retval, R, T = Resgistration3D(pt3dL, pt3dR)

    print(f"the number of detection is {numDetection}")

    return retval, R, T, pt2dL, pt2dR, pt3dL, pt3dR