import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import mediapipe as mp
from modules.utils.processing import augmentation_real
import json

ROOT_PATH = "/home/workplace/HOnnotate_OXR/dataset/segmentation"
SAVE_PATH = "/home/workplace/HOnnotate_OXR/dataset/dataForSeg"

image_cols, image_rows = 1080, 1920
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
numConsThreads = 1
w = 640
h = 480

segIndices = [1,5,9,13,17,1]
palmIndices = [0,1,5,9,13,17,0]
thumbIndices = [0,1,2,3,4]
indexIndices = [5,6,7,8]
middleIndices = [9,10,11,12]
ringIndices = [13,14,15,16]
pinkyIndices = [17,18,19,20]
lineIndices = [palmIndices, thumbIndices, indexIndices, middleIndices, ringIndices, pinkyIndices]

init_bbox = {}
init_bbox["mas"] = [400, 60, 1120, 960]
init_bbox["sub1"] = [360, 0, 1120, 960]
init_bbox["sub2"] = [640, 180, 640, 720]
init_bbox["sub3"] = [680, 180, 960, 720]

def extractBbox(idx_to_coord, image_rows, image_cols):
            bbox_width = 840
            bbox_height = 480
            # consider fixed size bbox
            x_min = min(idx_to_coord.values(), key=lambda x: x[0])[0]
            x_max = max(idx_to_coord.values(), key=lambda x: x[0])[0]
            y_min = min(idx_to_coord.values(), key=lambda x: x[1])[1]
            y_max = max(idx_to_coord.values(), key=lambda x: x[1])[1]

            x_avg = (x_min + x_max) / 2
            y_avg = (y_min + y_max) / 2

            x_min = max(0, x_avg - (bbox_width / 2))
            y_min = max(0, y_avg - (bbox_height / 2))

            if (x_min + bbox_width) > image_cols:
                x_min = image_cols - bbox_width
            if (y_min + bbox_height) > image_rows:
                y_min = image_rows - bbox_height

            bbox = [x_min, y_min, bbox_width, bbox_height]
            return bbox

def translateKpts(kps, img2bb):
        uv1 = np.concatenate((kps[:, :2], np.ones_like(kps[:, :1])), 1)
        kps[:, :2] = np.dot(img2bb, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
        return kps

def hand_mask():
    rgb_list = glob(os.path.join(ROOT_PATH, "*.jpg"))
    mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
    bbox = None
    for rgb_path in rgb_list:
        file_name = os.path.basename(rgb_path)
        if "mas" in file_name:
            bbox = init_bbox['mas']
        elif "sub1" in file_name:
            bbox = init_bbox['sub1']
        elif "sub2" in file_name:
            bbox = init_bbox['sub2']
        elif "sub3" in file_name:
            bbox = init_bbox['sub3']
        rgb = cv2.imread(rgb_path)
        image_rows, image_cols, _ = rgb.shape
        kps = np.empty((21, 3), dtype=np.float32)
        kps[:] = np.nan
        idx_to_coordinates = None
        rgb_copy = np.copy(rgb)
        rgb_copy = rgb_copy[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        results = mp_hand.process(cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == 21:
            print("hand detected")
            hand_landmark = results.multi_hand_landmarks[0]
            idx_to_coordinates = {}
            for idx_land, landmark in enumerate(hand_landmark.landmark):
                landmark_px = [landmark.x * bbox[2] + bbox[0], landmark.y * bbox[3] + bbox[1]]
                # landmark_px = [landmark.x* bbox[2], landmark.y* bbox[3]]
                if landmark_px:
                    # landmark_px has fliped x axis
                    idx_to_coordinates[idx_land] = [landmark_px[0], landmark_px[1]]

                    kps[idx_land, 0] = landmark_px[0]
                    kps[idx_land, 1] = landmark_px[1]
                    # save relative depth on z axis
                    kps[idx_land, 2] = landmark.z
        else:
            continue

        #temp
        bbox = extractBbox(idx_to_coordinates, image_rows, image_cols)
        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)
        kpts = translateKpts(np.copy(kps), img2bb_trans)
        cv2.imwrite(os.path.join(SAVE_PATH, "rgb", file_name), rgbCrop)
        rgbCopy = np.copy(rgbCrop)
        for lineIndex in lineIndices:
            for j in range(len(lineIndex)-1):
                point1 = np.int32(kpts[lineIndex[j], :2])
                point2 = np.int32(kpts[lineIndex[j+1], :2])
                cv2.line(rgbCopy, point1, point2, (255, 0, 0), 3)
        cv2.imwrite(os.path.join(SAVE_PATH, "results", file_name), rgbCopy)
        hsv_image = np.uint8(rgbCrop.copy())
        hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 150
        rgb_filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        seg_image = np.uint8(rgb_filtered.copy())
        mask = np.ones(seg_image.shape[:2], np.uint8) * 2
        for lineIndex in lineIndices:
            for j in range(len(lineIndex)-1):
                point1 = np.int32(kpts[lineIndex[j], :2])
                point2 = np.int32(kpts[lineIndex[j+1], :2])
                cv2.line(mask, point1, point2, cv2.GC_FGD, 1)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(seg_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask_hand = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        cv2.imwrite(os.path.join(SAVE_PATH, "hand_mask", file_name), mask_hand*255)

def obj_mask():
    rgb_list = glob(os.path.join(ROOT_PATH, "*.jpg"))
    objmask_path = "/home/workplace/HOnnotate_OXR/dataset/dataForSeg/obj_mask"

    for rgb_path in rgb_list:
        file_name = os.path.basename(rgb_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        contour_path = rgb_path + ".json"
        with open(contour_path, 'r') as f:
            contour = json.load(f)
        for annot in contour['annotations']:
            points = np.array(annot['points'])
            print(points.shape)
            cv2.drawContours(rgb, [points.astype(int)], -1, (255, 0, 0), -1)
        print(rgb.shape)
        cv2.imwrite(os.path.join(objmask_path, file_name), rgb)

def total_mask():
    rgb_list = glob(os.path.join(ROOT_PATH, "*.jpg"))
    mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
    bbox = None
    for rgb_path in rgb_list:
        file_name = os.path.basename(rgb_path)
        if "mas" in file_name:
            bbox = init_bbox['mas']
        elif "sub1" in file_name:
            bbox = init_bbox['sub1']
        elif "sub2" in file_name:
            bbox = init_bbox['sub2']
        elif "sub3" in file_name:
            bbox = init_bbox['sub3']
        rgb = cv2.imread(rgb_path)
        image_rows, image_cols, _ = rgb.shape
        kps = np.empty((21, 3), dtype=np.float32)
        kps[:] = np.nan
        idx_to_coordinates = None
        rgb_copy = np.copy(rgb)
        rgb_copy = rgb_copy[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        results = mp_hand.process(cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == 21:
            print("hand detected")
            hand_landmark = results.multi_hand_landmarks[0]
            idx_to_coordinates = {}
            for idx_land, landmark in enumerate(hand_landmark.landmark):
                landmark_px = [landmark.x * bbox[2] + bbox[0], landmark.y * bbox[3] + bbox[1]]
                # landmark_px = [landmark.x* bbox[2], landmark.y* bbox[3]]
                if landmark_px:
                    # landmark_px has fliped x axis
                    idx_to_coordinates[idx_land] = [landmark_px[0], landmark_px[1]]

                    kps[idx_land, 0] = landmark_px[0]
                    kps[idx_land, 1] = landmark_px[1]
                    # save relative depth on z axis
                    kps[idx_land, 2] = landmark.z
        else:
            continue

        #temp
        bbox = extractBbox(idx_to_coordinates, image_rows, image_cols)
        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)
        kpts = translateKpts(np.copy(kps), img2bb_trans)
        cv2.imwrite(os.path.join(SAVE_PATH, "rgb", file_name.replace('jpg', 'png')), rgbCrop)
        rgbCopy = np.copy(rgbCrop)
        for lineIndex in lineIndices:
            for j in range(len(lineIndex)-1):
                point1 = np.int32(kpts[lineIndex[j], :2])
                point2 = np.int32(kpts[lineIndex[j+1], :2])
                cv2.line(rgbCopy, point1, point2, (255, 0, 0), 3)
        cv2.imwrite(os.path.join(SAVE_PATH, "results", file_name), rgbCopy)
        hsv_image = np.uint8(rgbCrop.copy())
        hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 150
        rgb_filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        seg_image = np.uint8(rgb_filtered.copy())
        mask = np.ones(seg_image.shape[:2], np.uint8) * 2
        for lineIndex in lineIndices:
            for j in range(len(lineIndex)-1):
                point1 = np.int32(kpts[lineIndex[j], :2])
                point2 = np.int32(kpts[lineIndex[j+1], :2])
                cv2.line(mask, point1, point2, cv2.GC_FGD, 1)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(seg_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask_hand = np.where((mask==2)|(mask==0),0,1).astype('uint8') #HW

        cv2.imwrite(os.path.join(SAVE_PATH, "hand_mask", file_name), mask_hand*255)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        contour_path = rgb_path + ".json"
        with open(contour_path, 'r') as f:
            contour = json.load(f)
        obj_mask = np.zeros(rgb.shape)
        for annot in contour['annotations']:
            points = np.array(annot['points'])
            cv2.drawContours(obj_mask, [points.astype(int)], -1, (255, 0, 0), -1)
        obj_mask_crop, _, _, _, _, = augmentation_real(obj_mask, bbox, flip=False) #HW3
        obj_mask_crop = np.where(obj_mask_crop > 0, 1, 0).astype('uint8')
        obj_mask_crop = obj_mask_crop[:, :, 0]
        cv2.imwrite(os.path.join(SAVE_PATH, "obj_mask", file_name), obj_mask_crop*255)
        mask_hand = np.where(obj_mask_crop == 1, 0, mask_hand)
        final_mask = np.zeros(mask_hand.shape)
        final_mask = np.where(mask_hand == 1, 1, final_mask)
        final_mask = np.where(obj_mask_crop == 1, 2, final_mask)
        cv2.imwrite(os.path.join(SAVE_PATH, "mask", file_name.replace('jpg', 'png')), final_mask*125)


if __name__ == "__main__":
    total_mask()