import json
import numpy as np


def LoadCameraMatrix(path):
    result = {}
    order = []

    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if (not line): break
            if ("mas" in line) or ("master" in line):
                order.append("mas")
            elif "sub1" in line:
                order.append("sub1")
            elif "sub2" in line:
                order.append("sub2")
            elif "sub3" in line:
                order.append("sub3")

            if "[[" in line:
                intrinsicMatrices = np.empty((3,3))
                for i in range(3):
                    intrinsicMatrices[i,0] = float(line[2:14])
                    intrinsicMatrices[i,1] = float(line[15:27])
                    intrinsicMatrices[i,2] = float(line[28:40])
                    if (i != 2):
                        line = f.readline()
                    result[order[0]] = intrinsicMatrices
                del order[0]
    
    return result


def LoadDistortionParam(jsonFile):
    with open(jsonFile, "r") as camIntrinsic:
        camIntrinsicDict = json.load(camIntrinsic)

    params = camIntrinsicDict['CalibrationInformation']['Cameras'][1]['Intrinsics']['ModelParameters']

    DistCoeffs = np.array([params[4], params[5], params[13], params[12], *params[6:10]])

    return np.expand_dims(DistCoeffs,0)