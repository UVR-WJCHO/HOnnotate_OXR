import os
import glob
import json
import math
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from utils.loadParameters import LoadCameraMatrix, LoadDistortionParam
from utils.geometry import uv2xyz

cameras = ["mas", "sub1", "sub2", "sub3"]

intrinsicMatrices = LoadCameraMatrix("camera_parameters/220809_cameraInfo.txt")
distCoeffs = {}
distCoeffs["mas"] = LoadDistortionParam("camera_parameters/mas_intrinsic.json")
distCoeffs["sub1"] = LoadDistortionParam("camera_parameters/sub1_intrinsic.json")
distCoeffs["sub2"] = LoadDistortionParam("camera_parameters/sub2_intrinsic.json")
distCoeffs["sub3"] = LoadDistortionParam("camera_parameters/sub3_intrinsic.json")

imageDir = "hand_images/221115_fork_20_02/rgb"
depthDir = "hand_images/221115_fork_20_02/depth"
resultDir = "results/221115"
numHandJoints = 21
confidenceThreshold = 0.5
h = np.array([[0,0,0,1]])
numCameras = len(cameras)
numImages = min(math.ceil(len(glob.glob(os.path.join(depthDir, f"{camera}_*.png")))) for camera in cameras)

os.makedirs(resultDir, exist_ok=True)
os.makedirs(os.path.join(resultDir,"hand"), exist_ok=True)
os.makedirs(os.path.join(resultDir,"reprojected"), exist_ok=True)

with open(os.path.join(resultDir, "cameraParamsBA.json")) as json_file:
    cameraParams = json.load(json_file)
    cameraParams = {camera: np.array(cameraParams[camera]) for camera in cameraParams}

# rough bounding box for hand detection
boundingBox = {}
boundingBox["mas"] = {"x":[300,1000], "y":[0,720]}
boundingBox["sub1"] = {"x":[300,1000], "y":[0,720]}
boundingBox["sub2"] = {"x":[300,1000], "y":[0,720]}
boundingBox["sub3"] = {"x":[300,1000], "y":[0,720]}

palmIndices = [0,5,9,13,17,0]
thumbIndices = [0,1,2,3,4]
indexIndices = [5,6,7,8]
middleIndices = [9,10,11,12]
ringIndices = [13,14,15,16]
pinkyIndices = [17,18,19,20]
lineIndices = [palmIndices, thumbIndices, indexIndices, middleIndices, ringIndices, pinkyIndices]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

distortedHandDetection = np.zeros((numHandJoints, 2))
mediapipeDepth = np.zeros(numHandJoints)
detection4Dcamera = np.zeros((numHandJoints, 4))
resultDetection_uvd = np.zeros((numCameras, numCameras, numImages, numHandJoints, 3))
resultDetection_xyz = np.zeros((numCameras, numCameras, numImages, numHandJoints, 3))

hands = []
for _ in cameras:
    hand_model = mp_hands.Hands(max_num_hands=1, min_detection_confidence=confidenceThreshold)
    hands.append(hand_model)

for idx in tqdm(range(numImages)):
    # triangulationPoints = {}
    # distortedPoints = {}

    imageList = []
    imageCroppedList = []
    for i, camera in enumerate(cameras):
        # Read an image, flip it around y-axis for correct handedness output
        image = cv2.imread(os.path.join(imageDir, f"{camera}_{idx}.png"))

        # bounding box for current camera
        xRange = boundingBox[camera]["x"]
        yRange = boundingBox[camera]["y"]

        imageList.append(image)
        imageCroppedList.append(image[yRange[0]:yRange[1],xRange[0]:xRange[1]])

    # imageMerged1 = np.concatenate((imageCroppedList[0], imageCroppedList[1]), axis=1)
    # imageMerged2 = np.concatenate((imageCroppedList[2], imageCroppedList[3]), axis=1)
    # imageMerged = np.concatenate((imageMerged1, imageMerged2), axis=0)
    # cv2.imshow(os.path.join(resultDir, f"hand/{idx}.png"), imageMerged)
    # cv2.waitKey(0)
    imageDetectedList = []
    for i, camera in enumerate(cameras):
        # bounding box for current camera
        xRange = boundingBox[camera]["x"]
        yRange = boundingBox[camera]["y"]

        # Convert the BGR image to RGB before processing.
        results = hands[i].process(cv2.cvtColor(imageCroppedList[i], cv2.COLOR_BGR2RGB))

        annotatedImage = imageCroppedList[i].copy()

        if not results.multi_hand_landmarks:
            print(f"no hand detection in {idx}-th image of the {camera}")

            resultDetection_xyz[i, :, idx, :, :] = np.nan
            resultDetection_uvd[i, :, idx, :, :] = np.nan

            if len(cameras) == 4:
                imageMerged1 = np.concatenate((imageCroppedList[0], imageCroppedList[1]), axis=1)
                imageMerged2 = np.concatenate((imageCroppedList[2], imageCroppedList[3]), axis=1)
                imageMerged = np.concatenate((imageMerged1, imageMerged2), axis=0)
                cv2.imwrite(os.path.join(resultDir, f"reprojected/{camera}_{idx}.png"), imageMerged)
        else:
            handLandmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                annotatedImage,
                handLandmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # display confidence value
            score = results.multi_handedness[0].classification[0].score
            cv2.rectangle(annotatedImage, (0,0), (170,40), (0,0,0), -1)
            cv2.putText(annotatedImage, "%.4f"%score, (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)

            for k in range(numHandJoints):
                distortedHandDetection[k, 0] = handLandmarks.landmark[k].x * (xRange[1] - xRange[0]) + xRange[0]
                distortedHandDetection[k, 1] = handLandmarks.landmark[k].y * (yRange[1] - yRange[0]) + yRange[0]
                mediapipeDepth[k] = handLandmarks.landmark[k].z * (xRange[1] - xRange[0])

            # undistort normalized hand pose estimation points
            normalizedPoints = cv2.undistortPoints(distortedHandDetection, intrinsicMatrices[camera], distCoeffs[camera])
            normalizedPoints = normalizedPoints.squeeze().transpose()
            normalizedPoints = np.concatenate((normalizedPoints, np.ones((1,numHandJoints))),0)
            undistortedPoints = np.matmul(intrinsicMatrices[camera], normalizedPoints)[:2,:]

            # load kinect depth value
            depth = cv2.imread(os.path.join(depthDir, f"{camera}_{idx}.png"), -1)
            # use wrist joint as root for depth
            wrist2D = distortedHandDetection[0]
            fx = wrist2D[0]
            fy = wrist2D[1]
            x_ = np.uint16(np.floor(fx))
            y_ = np.uint16(np.floor(fy))
            wx = fx - x_
            wy = fy - y_
            try:
                value4pt = np.array([depth[y_,x_],depth[y_+1,x_],depth[y_,x_+1],depth[y_+1,x_+1]])
                weight4pt = np.array([(1-wx)*(1-wy), (1-wx)*(wy), (wx)*(1-wy), (wx)*(wy)])
            except:
                value4pt = np.array([0,0,0,0])

            if (value4pt == 0).any():
                print(f"no depth information in {idx}-th image of the {camera}")

                resultDetection_xyz[i, :, idx, :, :] = np.nan
                resultDetection_uvd[i, :, idx, :, :] = np.nan

                if len(cameras) == 4:
                    imageMerged1 = np.concatenate((imageCroppedList[0], imageCroppedList[1]), axis=1)
                    imageMerged2 = np.concatenate((imageCroppedList[2], imageCroppedList[3]), axis=1)
                    imageMerged = np.concatenate((imageMerged1, imageMerged2), axis=0)
                    cv2.imwrite(os.path.join(resultDir, f"reprojected/{camera}_{idx}.png"), imageMerged)
            else:
                # depth value of wrist joint from kinect
                wristDepth = np.dot(weight4pt,value4pt)

                for k in range(numHandJoints):
                    # "undistortedPoints" is normalized (it's in image plane)
                    detection4Dcamera[k,:] = np.concatenate((undistortedPoints[:,k], [1,1]))
                    # multiply the depth for 3d point in camera coordinate
                    detection4Dcamera[k,:] *= (wristDepth + mediapipeDepth[k])
                
                # homogeneous
                detection4Dcamera[:,3] = 1
                

                # (3d homogeneous point in camera coordinate) = (intrinsic) * (extrinsic) * (3d homogeneous point in world coordinate)
                projection = np.concatenate((intrinsicMatrices[camera] @ cameraParams[camera].reshape(3,4), h), 0)
                detection4Dworld = (np.linalg.inv(projection) @ detection4Dcamera.T).T

                # 2.5D detection in other reference cameras coordinates
                imageReprojectedList = []
                for j, camera2 in enumerate(cameras):
                    reprojectedImage = imageList[j].copy()
                    
                    # bounding box for current camera
                    xRange = boundingBox[camera2]["x"]
                    yRange = boundingBox[camera2]["y"]

                    # it is 3d homogeneous point in camera2 coordinate
                    detection4Dcamera2 = (cameraParams[camera2].reshape(3,4) @ detection4Dworld.T).T

                    # reprojected points (it is in image plane and distorted)
                    rvec, _ = cv2.Rodrigues(np.eye(3))
                    tvec = np.array([[0.,0.,0.]])
                    projectedPoints, _ = cv2.projectPoints(detection4Dcamera2[:,:3], rvec, tvec, intrinsicMatrices[camera], distCoeffs[camera])
                    projectedPoints = projectedPoints.squeeze()

                    # it is 3d homogeneous point in camera2 coordinate
                    resultDetection_xyz[i, j, idx, :, :] = detection4Dcamera2

                    # uv is reprojected, d is equal to z value
                    resultDetection_uvd[i, j, idx, :, :2] = projectedPoints
                    resultDetection_uvd[i, j, idx, :, 2] = detection4Dcamera2[:,2]

                    for x, pt2D in enumerate(projectedPoints):
                        cv2.circle(reprojectedImage, np.int32(pt2D), 3, (0,0,0), -1)
                    
                    for lineIndex in lineIndices:
                        for j in range(len(lineIndex)-1):
                            point1 = np.int32(projectedPoints[lineIndex[j]])
                            point2 = np.int32(projectedPoints[lineIndex[j+1]])
                            cv2.line(reprojectedImage, point1, point2, (255,255,255), 1)
                    
                    # bounding box for current camera
                    xRange = boundingBox[camera]["x"]
                    yRange = boundingBox[camera]["y"]

                    imageReprojectedList.append(reprojectedImage[yRange[0]:yRange[1],xRange[0]:xRange[1]])
                
                if len(cameras) == 4:
                    imageMerged1 = np.concatenate((imageReprojectedList[0], imageReprojectedList[1]), axis=1)
                    imageMerged2 = np.concatenate((imageReprojectedList[2], imageReprojectedList[3]), axis=1)
                    imageMerged = np.concatenate((imageMerged1, imageMerged2), axis=0)
                    cv2.imwrite(os.path.join(resultDir, f"reprojected/{camera}_{idx}.png"), imageMerged)
                
        imageDetectedList.append(annotatedImage)
    if len(cameras) == 4:
        imageMerged1 = np.concatenate((imageDetectedList[0], imageDetectedList[1]), axis=1)
        imageMerged2 = np.concatenate((imageDetectedList[2], imageDetectedList[3]), axis=1)
        imageMerged = np.concatenate((imageMerged1, imageMerged2), axis=0)
        cv2.imwrite(os.path.join(resultDir, f"hand/{idx}.png"), imageMerged)

result_xyz = {}
result_uvd = {}

for i, camera in enumerate(cameras):
    for j, camera2 in enumerate(cameras):
        result_xyz[f"{camera}_{camera2}"] = resultDetection_xyz[i,j].tolist()
        result_uvd[f"{camera}_{camera2}"] = resultDetection_uvd[i,j].tolist()

with open(os.path.join(resultDir, "handDetection_xyz.json"), "w") as fp:
    json.dump(result_xyz, fp)
with open(os.path.join(resultDir, "handDetection_uvd.json"), "w") as fp:
    json.dump(result_uvd, fp)

for hand in hands:
    hand.close()
