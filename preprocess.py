import os
import sys

sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models/slim'))

import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from absl import flags
from absl import app

# preprocess
import mediapipe as mp
from modules.utils.processing import augmentation_real
import numpy as np
from modules.utils.loadParameters import LoadCameraMatrix, LoadDistortionParam, LoadCameraParams

#temp
import matplotlib.pyplot as plt


# others
import cv2
from PIL import Image
import time
import json
import pickle
import copy
import tqdm
import torch
from torchvision.transforms.functional import to_tensor
import math


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230612', 'target db Name')   ## name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
flags.DEFINE_string('camID', 'mas', 'main target camera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
# camIDset = ['sub2', 'sub3']
FLAGS(sys.argv)

### Config ###
baseDir = os.path.join(os.getcwd(), 'dataset')
handResultDir = os.path.join(baseDir, FLAGS.db) + '_hand'
camResultDir = os.path.join(baseDir, FLAGS.db) + '_cam'
bgmModelPath = "/home/workplace/BackgroundMattingV2/model.pth"
"""
!pip install gdown -q
!gdown https://drive.google.com/uc?id=1-t9SO--H4WmP7wUl1tVNNeDkq47hjbv4 -O model.pth -q
"""

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
numConsThreads = 1
w = 640
h = 480

palmIndices = [0,5,9,13,17,0]
thumbIndices = [0,1,2,3,4]
indexIndices = [5,6,7,8]
middleIndices = [9,10,11,12]
ringIndices = [13,14,15,16]
pinkyIndices = [17,18,19,20]
lineIndices = [palmIndices, thumbIndices, indexIndices, middleIndices, ringIndices, pinkyIndices]

### Manual Flags (remove after debug) ###
flag_preprocess = True
flag_segmentation = True

class loadDataset():
    def __init__(self, db, seq):
        self.seq = seq

        self.dbDir = os.path.join(baseDir, db, seq)
        self.bgDir = os.path.join(baseDir, FLAGS.db) + '_background'
        self.handDir = handResultDir

        self.rgbDir = os.path.join(self.dbDir, 'rgb')
        self.depthDir = os.path.join(self.dbDir, 'depth')
        self.rgbBgDir = os.path.join(self.bgDir, 'rgb')
        self.depthBgDir = os.path.join(self.bgDir, 'depth')

        if not os.path.exists(os.path.join(self.dbDir, 'rgb_crop')):
            os.mkdir(os.path.join(self.dbDir, 'rgb_crop'))
            for camID in camIDset:
                os.mkdir(os.path.join(self.dbDir, 'rgb_crop', camID))
        if not os.path.exists(os.path.join(self.dbDir, 'depth_crop')):
            os.mkdir(os.path.join(self.dbDir, 'depth_crop'))
            for camID in camIDset:
                os.mkdir(os.path.join(self.dbDir, 'depth_crop', camID))
        if not os.path.exists(os.path.join(self.dbDir, 'meta')):
            os.mkdir(os.path.join(self.dbDir, 'meta'))
            for camID in camIDset:
                os.mkdir(os.path.join(self.dbDir, 'meta', camID))
        
        if not os.path.exists(os.path.join(self.dbDir, 'segmentation')):
            os.mkdir(os.path.join(self.dbDir, 'segmentation'))
            for camID in camIDset:
                os.mkdir(os.path.join(self.dbDir, 'segmentation', camID))
                os.mkdir(os.path.join(self.dbDir, 'segmentation', camID, 'visualization'))
                os.mkdir(os.path.join(self.dbDir, 'segmentation', camID, 'raw_seg_results'))
        #temp
        if not os.path.exists(os.path.join(self.dbDir, 'masked_rgb')):
            os.mkdir(os.path.join(self.dbDir, 'masked_rgb'))
            for camID in camIDset:
                os.mkdir(os.path.join(self.dbDir, 'masked_rgb', camID))
                os.mkdir(os.path.join(self.dbDir, 'masked_rgb', camID, 'bg'))

        self.rgbCropDir = None
        self.depthCropDir = None
        self.metaDir = None

        self.bbox_width = 640
        self.bbox_height = 480
<<<<<<< HEAD
        # 매뉴얼 하게 초기 bbox를 찾아서 설정해줌
=======
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
        self.temp_bbox = {}
        self.temp_bbox["mas"] = [650, 280, self.bbox_width, self.bbox_height]
        self.temp_bbox["sub1"] = [750, 300, self.bbox_width, self.bbox_height]
        self.temp_bbox["sub2"] = [500, 150, self.bbox_width, self.bbox_height]
        self.temp_bbox["sub3"] = [650, 200, self.bbox_width, self.bbox_height]
        self.prev_bbox = None
        self.wrist_px = None
<<<<<<< HEAD

        # intrinsics, dist_coeffs, extrinsics = LoadCameraParams(os.path.join(camResultDir, "cameraParams.json"))
        # self.intrinsics = intrinsics
        # self.distCoeffs = dist_coeffs
=======
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
        self.intrinsics = LoadCameraMatrix(os.path.join(camResultDir, "230612_cameraInfo.txt"))
        self.distCoeffs = {}
        self.distCoeffs["mas"] = LoadDistortionParam(os.path.join(camResultDir, "mas_intrinsic.json"))
        self.distCoeffs["sub1"] = LoadDistortionParam(os.path.join(camResultDir, "sub1_intrinsic.json"))
        self.distCoeffs["sub2"] = LoadDistortionParam(os.path.join(camResultDir, "sub2_intrinsic.json"))
        self.distCoeffs["sub3"] = LoadDistortionParam(os.path.join(camResultDir, "sub3_intrinsic.json"))

        self.intrinsic_undistort = os.path.join(camResultDir, FLAGS.db + "_cameraInfo_undistort.txt")
        self.prev_cam_check = None
        if os.path.isfile(self.intrinsic_undistort):
            self.flag_save = False
        else:
            self.flag_save = True
            with open(self.intrinsic_undistort, "w") as f:
                print("creating undistorted intrinsic of each cam")



    def __len__(self):
        return len(os.listdir(self.rgbDir))
    
    def init_cam(self, camID, threshold=0.3):
        self.rgbCropDir = os.path.join(self.dbDir, 'rgb_crop', camID)
        self.depthCropDir = os.path.join(self.dbDir, 'depth_crop', camID)
        self.metaDir = os.path.join(self.dbDir, 'meta', camID)
        segDir = os.path.join(self.dbDir, 'segmentation', camID)
        self.segVisDir = os.path.join(segDir, 'visualization')
        self.segResDir = os.path.join(segDir, 'raw_seg_results')

        #temp
        self.maskedRgbDir = os.path.join(self.dbDir, 'masked_rgb', camID)
        self.croppedBgDir = os.path.join(self.dbDir, 'masked_rgb', camID, 'bg')
        
        self.debugDir = os.path.join(self.dbDir, 'debug')
        self.mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=threshold) 
        self.K = self.intrinsics[camID]
        self.dist = self.distCoeffs[camID]

        self.prev_bbox = self.temp_bbox[camID]

    def getItem(self, idx, camID='mas'):
        # camID : mas, sub1, sub2, sub3
        imgName = str(camID) + '_' + str(idx) + '.png'

        rgbPath = os.path.join(self.rgbDir, imgName)
        depthPath = os.path.join(self.depthDir, imgName)

        assert os.path.exists(rgbPath), 'rgb image does not exist'
        assert os.path.exists(depthPath), 'depth image does not exist'

        rgb = cv2.imread(rgbPath)
        depth = cv2.imread(depthPath, cv2.IMREAD_ANYDEPTH)

        return (rgb, depth)
    
<<<<<<< HEAD
    def getBg(self, camID='mas'):
        bgName = str(camID) + '_1' + '.png'
        rgbBgPath = os.path.join(self.rgbBgDir, bgName)
        depthBgPath = os.path.join(self.depthBgDir, bgName)

        assert os.path.exists(rgbBgPath), 'rgb background image does not exist'
        assert os.path.exists(depthBgPath), 'depth background image does not exist'

        rgbBg = cv2.imread(rgbBgPath)
        depthBg = cv2.imread(depthBgPath, cv2.IMREAD_ANYDEPTH)

        return (rgbBg, depthBg)

=======
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
    def undistort(self, images, camID):
        rgb, depth = images
        image_cols, image_rows = rgb.shape[:2]
        self.new_camera, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (image_rows, image_cols), 1, (image_rows, image_cols))
        rgb = cv2.undistort(rgb, self.K, self.dist, None, self.new_camera)
        depth = cv2.undistort(depth, self.K, self.dist, None, self.new_camera)

        # print(self.new_camera)
        # exit(0)
        if self.prev_cam_check != camID and self.flag_save:
            self.prev_cam_check = camID

            with open(self.intrinsic_undistort, "a") as f:
                intrinsic_undistort = str(np.copy(self.new_camera))
                f.write(intrinsic_undistort)
                f.write("\n")

            if camID == camIDset[-1]:
                self.flag_save = False

        return (rgb, depth)
    
    def procImg(self, images):
        rgb, depth = images
        image_rows, image_cols, _ = rgb.shape
<<<<<<< HEAD
        idx_to_coordinates = None
        kps = np.empty((21, 3), dtype=np.float32)
=======
        kps = np.empty((21, 2), dtype=np.float32)
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
        kps[:] = np.nan
        idx_to_coordinates = None

<<<<<<< HEAD
        # 반복해서 검출 시도
        for _ in range(2):
=======
        for _ in range(5):
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
            results = self.mp_hand.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == 21:
                hand_landmark = results.multi_hand_landmarks[0]
                idx_to_coordinates = {}
<<<<<<< HEAD
                # wristDepth = depth[
                #     int(hand_landmark.landmark[0].y * image_rows), int(hand_landmark.landmark[0].x * image_cols)]
                for idx_land, landmark in enumerate(hand_landmark.landmark):
                    landmark_px = [landmark.x * image_cols, landmark.y * image_rows]
=======
                wristDepth = depth[int(hand_landmark.landmark[0].y*image_rows), int(hand_landmark.landmark[0].x*image_cols)]
                for idx_land, landmark in enumerate(hand_landmark.landmark):
                    landmark_px = [landmark.x*image_cols, landmark.y*image_rows]
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
                    if landmark_px:
                        # landmark_px has fliped x axis
                        orig_x = landmark_px[0]
                        idx_to_coordinates[idx_land] = [orig_x, landmark_px[1]]

                        kps[idx_land, 0] = landmark_px[0]
                        kps[idx_land, 1] = landmark_px[1]
<<<<<<< HEAD
                        # save relative depth on z axis
                        kps[idx_land, 2] = landmark.z


            # 전체 이미지에서 손 검출이 안되는 경우 이전 bbox로 크롭한 후에 다시 검출
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks[0].landmark) != 21 or np.any(
                    np.isnan(kps)):
                bbox = self.prev_bbox
                rgb_crop = rgb[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
=======
                        # 3d keypoints from depth image and mediapipe
                        # if wristDepth > 0:
                        #     kps[idx_land, 0] = landmark_px[0]
                        #     kps[idx_land, 1] = landmark_px[1]
                        #     kps[idx_land, 2] = landmark.z + wristDepth
                        # else:
                        #     print("wristDepth is 0")

            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks[0].landmark) != 21 or np.any(np.isnan(kps)):
                bbox = self.prev_bbox
                rgb_crop = rgb[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
                results = self.mp_hand.process(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    idx_to_coordinates = {}
<<<<<<< HEAD
                    # wristDepth = depth[int(hand_landmarks.landmark[0].y * bbox[3] + bbox[1]), int(
                    #     hand_landmarks.landmark[0].x * bbox[2] + bbox[0])]
                    for idx_land, landmarks in enumerate(hand_landmarks.landmark):
                        landmarks_px = [int(landmarks.x * bbox[2] + bbox[0]), int(landmarks.y * bbox[3] + bbox[1])]
                        if landmarks_px:
                            idx_to_coordinates[idx_land] = landmarks_px

                            kps[idx_land, 0] = landmarks_px[0]
                            kps[idx_land, 1] = landmarks_px[1]
                            # save relative depth on z axis
                            kps[idx_land, 2] = landmarks.z

            # 손이 나왔을 경우 반복 멈춤
            if not np.any(np.isnan(kps)):
                break

        idx_to_coord = idx_to_coordinates

        # 크롭 후에도 검출이 안된다면 이전 키포인트를 그대로 사용
=======
                    wristDepth = depth[int(hand_landmarks.landmark[0].y*bbox[3]+bbox[1]), int(hand_landmarks.landmark[0].x*bbox[2]+bbox[0])]
                    for idx_land, landmarks in enumerate(hand_landmarks.landmark):
                        landmarks_px = [int(landmarks.x*bbox[2]+bbox[0]), int(landmarks.y*bbox[3]+bbox[1])]
                        if landmarks_px:
                            idx_to_coordinates[idx_land] = landmarks_px
                        
                            kps[idx_land, 0] = landmarks_px[0]
                            kps[idx_land, 1] = landmarks_px[1]
                            # if wristDepth > 0:
                            #     kps[idx_land, 0] = landmarks_px[0]
                            #     kps[idx_land, 1] = landmarks_px[1]
                            #     kps[idx_land, 2] = wristDepth + landmarks.z
                            # else:
                            #     print("wristDepth is 0")
            
            if not np.any(np.isnan(kps)):
                break
        
        idx_to_coord = idx_to_coordinates

>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
        if idx_to_coord is not None:
            self.prev_idx_to_coord = idx_to_coord
            self.prev_kps = kps
        else:
            idx_to_coord = self.prev_idx_to_coord
            kps = self.prev_kps

        bbox = self.extractBbox(idx_to_coord, image_rows, image_cols)
        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)
        depthCrop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        depthMask = np.where(depth < 3, 1, 0).astype(np.uint8)[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

<<<<<<< HEAD
        procImgSet = [rgbCrop, depthCrop]
        self.prev_bbox = copy.deepcopy(bbox)

        return bbox, img2bb_trans, bb2img_trans, procImgSet, kps


=======
        procImgSet = [rgbCrop, depthCrop, depthMask]
        self.prev_bbox = copy.deepcopy(bbox)

        return [bbox, img2bb_trans, bb2img_trans, procImgSet, kps]
    
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
    def extractBbox(self, idx_to_coord, image_rows, image_cols):
            # consider fixed size bbox
            x_min = min(idx_to_coord.values(), key=lambda x: x[0])[0]
            x_max = max(idx_to_coord.values(), key=lambda x: x[0])[0]
            y_min = min(idx_to_coord.values(), key=lambda x: x[1])[1]
            y_max = max(idx_to_coord.values(), key=lambda x: x[1])[1]

            x_avg = (x_min + x_max) / 2
            y_avg = (y_min + y_max) / 2

            x_min = max(0, x_avg - (self.bbox_width / 2))
            y_min = max(0, y_avg - (self.bbox_height / 2))

            if (x_min + self.bbox_width) > image_cols:
                x_min = image_cols - self.bbox_width
            if (y_min + self.bbox_height) > image_rows:
                y_min = image_rows - self.bbox_height

            bbox = [x_min, y_min, self.bbox_width, self.bbox_height]
            return bbox

    def translateKpts(self, kps, img2bb):
        uv1 = np.concatenate((kps[:, :2], np.ones_like(kps[:, :1])), 1)
        kps[:, :2] = np.dot(img2bb, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
        return kps
    
    def backgroundMatting(self, imgs, bgs, bbox):
        src, depth = imgs #non cropped image
        bgr, _ = bgs #non cropped backbround
        cv2.imwrite(os.path.join(self.croppedBgDir, 'bg.png'), bgr)
        model = torch.jit.load(bgmModelPath).cuda().eval()
        src = src[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        bgr = bgr[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src = Image.fromarray(src)
        bgr = Image.fromarray(bgr)
        src = to_tensor(src).cuda().unsqueeze(0)
        bgr = to_tensor(bgr).cuda().unsqueeze(0)
        if src.size(2) <= 2048 and src.size(3) <= 2048:
            model.backbone_scale = 1/4
            model.refine_sample_pixels = 80_000
        else:
            model.backbone_scale = 1/8
            model.refine_sample_pixels = 320_000
        pha, fgr = model(src, bgr)[:2]
        mask = pha[0].permute(1, 2, 0).cpu().numpy()*255
        com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
        com = com.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
        com = cv2.cvtColor(com.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # return mask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])], com[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        return mask, com
    
    def segmenation(self, camID, idx, procImgSet, kps):
        rgb, _, depth_mask = procImgSet

<<<<<<< HEAD
        rgb, _, matting, mattedRgb = procImgSet
        seg_image = np.uint8(mattedRgb.copy())
        # seg_image = np.uint8(rgb.copy())
=======
        seg_image = np.uint8(rgb.copy())
>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
        mask = np.ones(seg_image.shape[:2], np.uint8) * 2
        for lineIndex in lineIndices:
            for j in range(len(lineIndex)-1):
                point1 = np.int32(kps[lineIndex[j], :2])
                point2 = np.int32(kps[lineIndex[j+1], :2])
                cv2.line(mask, point1, point2, 1, 1)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(seg_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        imgName = str(camID) + '_' + format(idx, '04') + '.png'
        cv2.imwrite(os.path.join(self.segResDir, imgName), mask * 255)
        cv2.imwrite(os.path.join(self.segVisDir, imgName), seg_image*mask[:,:,np.newaxis])

    def postProcess(self, idx, procImgSet, bb, img2bb, bb2img, kps, processed_kpts, camID='mas'):
        imgName = str(camID) + '_' + format(idx, '04') + '.png'
        cv2.imwrite(os.path.join(self.rgbCropDir, imgName), procImgSet[0])
        cv2.imwrite(os.path.join(self.depthCropDir, imgName), procImgSet[1])
        #temp
        cv2.imwrite(os.path.join(self.maskedRgbDir, imgName), procImgSet[2])

        meta_info = {'bb': bb, 'img2bb': np.float32(img2bb),
                     'bb2img': np.float32(bb2img), 'kpts': np.float32(kps), 'kpts_crop': np.float32(processed_kpts)}

        metaName = str(camID) + '_' + format(idx, '04') + '.pkl'
        jsonPath = os.path.join(self.metaDir, metaName)
        with open(jsonPath, 'wb') as f:
            pickle.dump(meta_info, f, pickle.HIGHEST_PROTOCOL)

################# depth scale value need to be update #################
def main(argv):
    ### Setup ###
    rootDir = os.path.join(baseDir, FLAGS.db)
    lenDBTotal = len(os.listdir(rootDir))

    ### Hand pose initialization(mediapipe) ###
    '''
    [TODO]
        - consider two-hand situation (currently assume single hand detection)
    '''

    ### Preprocess ###
    if flag_preprocess:
        print("---------------start preprocess---------------")
        for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
            db = loadDataset(FLAGS.db, seqName)

            # db includes data for [mas, sub1, sub2, sub3]
            for camID in camIDset:
                if camID == 'sub2':
                    continue
                db.init_cam(camID)
                bgs = db.getBg(camID)
                bgs = db.undistort(bgs, camID)
                pbar = tqdm.tqdm(range(int(len(db) / 4)))
                for idx in pbar:
                    if idx < 120 or idx > 140:
                        continue
                    images = db.getItem(idx, camID=camID)
                    images = db.undistort(images, camID)

                    procResult = db.procImg(images) #bb, img2bb, bb2img, procImgSet, kps
                    bb, img2bb, bb2img, procImgSet, kps = procResult
                    procKps = db.translateKpts(np.copy(kps), img2bb)
<<<<<<< HEAD
                    matting, mattedRgb = db.backgroundMatting(images, bgs, bb)
                    procImgSet.append(matting)
                    procImgSet.append(mattedRgb)
=======

                    if np.any(np.isnan(kps)) or np.any(np.isnan(procKps)):
                        print("nan detected")
                        # bb, img2bb, bb2img, procImgSet, kps = prevResult

>>>>>>> 12204fdc3787a0ae75f6000e6405fac60c703dde
                    db.postProcess(idx, procImgSet, bb, img2bb, bb2img, kps, procKps, camID=camID)
                    if flag_segmentation:
                        db.segmenation(camID, idx, procImgSet, procKps)
                    pbar.set_description("(%s in %s) : (cam %s, idx %s) in %s" % (seqIdx, lenDBTotal, camID, idx, seqName))
        print("---------------end preprocess---------------")

if __name__ == '__main__':
    app.run(main)