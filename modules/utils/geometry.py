import numpy as np


def GetDepths(depth,points2D, undistortedPoints2D):
    ds = []
    retval = []
    for i, pt2D in enumerate(points2D):
        fx = pt2D[0]
        fy = pt2D[1]
        x_ = np.uint16(np.floor(fx))
        y_ = np.uint16(np.floor(fy))
        wx = fx - x_
        wy = fy - y_

        value4pt = np.array([depth[y_,x_],depth[y_+1,x_],depth[y_,x_+1],depth[y_+1,x_+1]])
        weight4pt = np.array([(1-wx)*(1-wy), (1-wx)*(wy), (wx)*(1-wy), (wx)*(wy)])

        if (value4pt == 0).any():
            retval.append(False)
        else:
            retval.append(True)

        d = np.dot(weight4pt,value4pt)
        ds.append([undistortedPoints2D[i,0]*d,undistortedPoints2D[i,1]*d,d])

    return retval, np.array(ds)


def uv2xyz(camMat,depth,xy,xyUndistorted):
    valz, xyz = GetDepths(depth, xy, xyUndistorted)

    c_intrinsic_inv = np.linalg.inv(camMat)

    XYZ = np.matmul(c_intrinsic_inv,xyz.T).T

    return valz, XYZ