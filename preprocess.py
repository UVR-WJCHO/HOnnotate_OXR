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
from utils.lossUtils import *
import csv
import datetime as dt
from config import *

# multiprocessing
from tqdm_multiprocess import TqdmMultiProcessPool


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230824', 'target db Name')   ## name ,default, help
flags.DEFINE_string('cam_db', '230824_cam', 'target cam db Name')   ## name ,default, help

flags.DEFINE_string('camID', 'mas', 'main target camera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

### Config ###
baseDir = os.path.join(os.getcwd(), 'dataset')
bgmModelPath = os.path.join(os.getcwd(), 'model.pth')
"""
Background matting을 위해 pretrained model 다운
!pip install gdown -q
!gdown https://drive.google.com/uc?id=1-t9SO--H4WmP7wUl1tVNNeDkq47hjbv4 -O model.pth -q
"""
camResultDir = os.path.join(baseDir, FLAGS.cam_db)


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
    def __init__(self, db, seq, trial):
        self.seq = seq # 230612_S01_obj_01_grasp_01
        self.db = db
        self.subject_id = seq.split('_')[1][1:]
        self.obj_id = seq.split('_')[3]
        self.grasp_id = seq.split('_')[5]
        self.trial = trial
        self.trial_num = trial.split('_')[1]

        self.dbDir = os.path.join(baseDir, db, seq, trial)
        # self.bgDir = os.path.join(baseDir, FLAGS.db) + '_background'

        self.rgbDir = os.path.join(self.dbDir, 'rgb')
        self.depthDir = os.path.join(self.dbDir, 'depth')
        # self.rgbBgDir = os.path.join(self.bgDir, 'rgb')
        # self.depthBgDir = os.path.join(self.bgDir, 'depth')

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

        if not os.path.exists(os.path.join(self.dbDir, 'annotation')):
            os.mkdir(os.path.join(self.dbDir, 'annotation'))
        
        # if not os.path.exists(os.path.join(self.dbDir, 'masked_rgb')):
        #     os.mkdir(os.path.join(self.dbDir, 'masked_rgb'))
        #     for camID in camIDset:
        #         os.mkdir(os.path.join(self.dbDir, 'masked_rgb', camID))
        #         os.mkdir(os.path.join(self.dbDir, 'masked_rgb', camID, 'bg'))

        # if not os.path.exists(os.path.join(self.dbDir, 'visualize')):
        #     os.mkdir(os.path.join(self.dbDir, 'visualize'))
        # self.debug_vis = os.path.join(self.dbDir, 'visualize')
            
        self.rgbCropDir = None
        self.depthCropDir = None
        self.metaDir = None

        self.bbox_width = 640
        self.bbox_height = 480
        # 매뉴얼 하게 초기 bbox를 찾아서 설정해줌
        self.temp_bbox = {}
        self.temp_bbox["mas"] = [650, 280, self.bbox_width, self.bbox_height]
        self.temp_bbox["sub1"] = [750, 300, self.bbox_width, self.bbox_height]
        self.temp_bbox["sub2"] = [500, 150, self.bbox_width, self.bbox_height]
        self.temp_bbox["sub3"] = [650, 200, self.bbox_width, self.bbox_height]
        self.prev_bbox = None
        self.wrist_px = None
        # Data 230612
        if FLAGS.db == "230612":
            self.intrinsics = LoadCameraMatrix(os.path.join(camResultDir, "%s_cameraInfo.txt"%FLAGS.db))
            self.distCoeffs = {}
            self.distCoeffs["mas"] = LoadDistortionParam(os.path.join(camResultDir, "mas_camInfo.json"))
            self.distCoeffs["sub1"] = LoadDistortionParam(os.path.join(camResultDir, "sub1_camInfo.json"))
            self.distCoeffs["sub2"] = LoadDistortionParam(os.path.join(camResultDir, "sub2_camInfo.json"))
            self.distCoeffs["sub3"] = LoadDistortionParam(os.path.join(camResultDir, "sub3_camInfo.json"))
            with open(os.path.join(camResultDir, "cameraParamsBA.json")) as json_file:
                self.extrinsics = json.load(json_file)
        else: # Data 2308~
            intrinsics, dist_coeffs, extrinsics = LoadCameraParams(os.path.join(self.cam_path, "cameraParams.json"))
            self.intrinsics = intrinsics
            self.distCoeffs = dist_coeffs
            self.extrinsics = extrinsics

        self.intrinsic_undistort = os.path.join(camResultDir, "cameraInfo_undistort.txt")
        self.prev_cam_check = None
        if os.path.isfile(self.intrinsic_undistort):
            self.flag_save = False
        else:
            self.flag_save = True
            with open(self.intrinsic_undistort, "w") as f:
                print("creating undistorted intrinsic of each cam")

        self.prev_kps = None
        self.prev_idx_to_coord = None

    def __len__(self):
        return len(os.listdir(os.path.join(self.rgbDir, 'mas')))

    def init_cam(self, camID):
        self.camID = camID
        self.rgbCropDir = os.path.join(self.dbDir, 'rgb_crop', camID)
        self.depthCropDir = os.path.join(self.dbDir, 'depth_crop', camID)
        self.metaDir = os.path.join(self.dbDir, 'meta', camID)
        segDir = os.path.join(self.dbDir, 'segmentation', camID)
        self.segVisDir = os.path.join(segDir, 'visualization')
        self.segResDir = os.path.join(segDir, 'raw_seg_results')

        #for background matting
        self.maskedRgbDir = os.path.join(self.dbDir, 'masked_rgb', camID)
        self.croppedBgDir = os.path.join(self.dbDir, 'masked_rgb', camID, 'bg')
        
        self.debugDir = os.path.join(self.dbDir, 'debug')
        self.K = self.intrinsics[camID]
        self.dist = self.distCoeffs[camID]

        self.prev_bbox = self.temp_bbox[camID]

    def init_info(self):
        if self.camID == 'mas':
            self.annotDir = os.path.join(self.dbDir, 'annotation')
            #====================================================
            # Dummy data 임시
            input = "./dummy_annotation_meta.csv"

            with open(input, "r", encoding="utf-8", newline="") as r:
                f = csv.reader(r, delimiter=",", skipinitialspace=True)
                csv_dict = {}
                for row in f:
                    csv_dict[row[0]] = row[1]
            #====================================================
            #Sequence 내 공동정보 저장
            with open(os.path.join(os.getcwd(), 'info_template.json')) as r:
                info = json.load(r)
            #info keys 'info', 'actor', 'kinect_camera', 'infrared_camera', 'images', 'object', 'calibration', 'annotations', 'Mesh'
            # 기록해 놓은 actor 정보 받아오기
            info['info']['data_created'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            info['actor']['id'] = csv_dict['actor.id']
            info['actor']['sex'] = csv_dict['actor.sex']
            info['actor']['age'] = csv_dict['actor.age']
            info['actor']['height'] = csv_dict['actor.height']
            info['actor']['handsize'] = csv_dict['actor.handsize']
            #카메라 정보 받아오기
            info['kinect_camera']['name'] = csv_dict['kinect_camera.name']
            info['kinect_camera']['time'] = len(os.listdir(os.path.join(self.rgbDir, 'mas'))) / 30.0 #TODO 프레임레이트 확인 필요
            info['kinect_camera']['id'] = csv_dict['kinect_camera.id']
            # 고정값
            info['kinect_camera']['height'] = "1080"
            info['kinect_camera']['width'] = "1920"

            info['infrared_camera']['name'] = csv_dict['infrared_camera.name']
            info['infrared_camera']['time'] = csv_dict['infrared_camera.time']
            info['infrared_camera']['id'] = csv_dict['infrared_camera.id']
            # 고정값
            info['infrared_camera']['height'] = "1080"
            info['infrared_camera']['width'] = "1920"
            info['infrared_camera']['frame'] = "60"
            info['infrared_camera']['resolution'] = "0.14"
            info['infrared_camera']['optics'] = "S"

            info['calibration']['sderror'] = "0.05" #TODO 확인 필요
            info['calibration']['extrinsic'] = self.extrinsics
            info['calibration']['intrinsic'] = self.intrinsics

            info['object']['id'] = self.obj_id
            info['object']['name'] = OBJType(int(self.obj_id)).name
            info['object']['marker_count'] = csv_dict['object.marker_count']
            info['object']['markers_data'] = csv_dict['markers_data']
            info['object']['pose_data'] = csv_dict['pose_data']
            self.info = info
        else:
            self.info = None

    def getItem(self, idx):
        # camID : mas, sub1, sub2, sub3
        rgbName = str(self.camID) + '/' + str(self.camID) + '_' + str(idx) + '.jpg'
        depthName = str(self.camID) + '/' + str(self.camID) + '_' + str(idx) + '.png'

        rgbPath = os.path.join(self.rgbDir, rgbName)
        depthPath = os.path.join(self.depthDir, depthName)

        assert os.path.exists(rgbPath), f'{rgbPath} rgb image does not exist'
        assert os.path.exists(depthPath), f'{depthPath} depth image does not exist'

        rgb = cv2.imread(rgbPath)
        depth = cv2.imread(depthPath, cv2.IMREAD_ANYDEPTH)
        if self.info != None:
            for cam in camIDset:
                self.info['images']['file_name'].append(os.path.join("rgb", str(cam) + '/' + str(cam) + '_' + str(idx) + '.jpg'))
                self.info['images']['file_name'].append(os.path.join("depth", str(cam) + '/' + str(cam) + '_' + str(idx) + '.png'))
            self.info['images']['id'] = str(self.db) + str(self.subject_id) + str(self.obj_id) + str(self.grasp_id) + str(self.trial_num)
            self.info['images']['width'] = rgb.shape[1]
            self.info['images']['height'] = rgb.shape[0]
            self.info['images']['data_created'] = dt.datetime.fromtimestamp(os.path.getctime(rgbPath)).strftime('%Y-%m-%d %H:%M')
            self.info['frame_num'] = idx

        return (rgb, depth)
    
    def getBg(self):
        bgName = str(self.camID) + '_1' + '.png'
        rgbBgPath = os.path.join(self.rgbBgDir, bgName)
        depthBgPath = os.path.join(self.depthBgDir, bgName)

        assert os.path.exists(rgbBgPath), 'rgb background image does not exist'
        assert os.path.exists(depthBgPath), 'depth background image does not exist'

        rgbBg = cv2.imread(rgbBgPath)
        depthBg = cv2.imread(depthBgPath, cv2.IMREAD_ANYDEPTH)

        return (rgbBg, depthBg)

    def undistort(self, images):
        rgb, depth = images
        image_cols, image_rows = rgb.shape[:2]
        self.new_camera, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (image_rows, image_cols), 1, (image_rows, image_cols))
        rgb = cv2.undistort(rgb, self.K, self.dist, None, self.new_camera)
        depth = cv2.undistort(depth, self.K, self.dist, None, self.new_camera)

        # print(self.new_camera)
        # exit(0)
        if self.prev_cam_check != self.camID and self.flag_save:
            self.prev_cam_check = self.camID

            with open(self.intrinsic_undistort, "a") as f:
                intrinsic_undistort = str(np.copy(self.new_camera))
                f.write(intrinsic_undistort)
                f.write("\n")

            if self.camID == camIDset[-1]:
                self.flag_save = False

        return (rgb, depth)
    
    def procImg(self, images, mp_hand):
        rgb, depth = images
        image_rows, image_cols, _ = rgb.shape
        kps = np.empty((21, 3), dtype=np.float32)
        kps[:] = np.nan
        idx_to_coordinates = None

        # 반복해서 검출 시도
        for _ in range(2):
            results = mp_hand.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == 21:
                hand_landmark = results.multi_hand_landmarks[0]
                idx_to_coordinates = {}
                # wristDepth = depth[
                #     int(hand_landmark.landmark[0].y * image_rows), int(hand_landmark.landmark[0].x * image_cols)]
                for idx_land, landmark in enumerate(hand_landmark.landmark):
                    landmark_px = [landmark.x * image_cols, landmark.y * image_rows]
                    if landmark_px:
                        # landmark_px has fliped x axis
                        orig_x = landmark_px[0]
                        idx_to_coordinates[idx_land] = [orig_x, landmark_px[1]]

                        kps[idx_land, 0] = landmark_px[0]
                        kps[idx_land, 1] = landmark_px[1]
                        # save relative depth on z axis
                        kps[idx_land, 2] = landmark.z


            # 전체 이미지에서 손 검출이 안되는 경우 이전 bbox로 크롭한 후에 다시 검출
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks[0].landmark) != 21 or np.any(
                    np.isnan(kps)):
                bbox = self.prev_bbox
                rgb_crop = rgb[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                results = mp_hand.process(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    idx_to_coordinates = {}
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
        if idx_to_coord is not None:
            self.prev_idx_to_coord = idx_to_coord
            self.prev_kps = kps
        else:
            if self.prev_kps is None:
                return [None]
            else:
                idx_to_coord = self.prev_idx_to_coord
                kps = self.prev_kps

        bbox = self.extractBbox(idx_to_coord, image_rows, image_cols)
        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)
        depthCrop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        depthMask = np.where(depth < 3, 1, 0).astype(np.uint8)[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        procImgSet = [rgbCrop, depthCrop]
        self.prev_bbox = copy.deepcopy(bbox)

        return [bbox, img2bb_trans, bb2img_trans, procImgSet, kps]


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
    
    def segmenation(self, idx, procImgSet, kps):
        rgb, _ = procImgSet

        # rgb, _, matting, mattedRgb = procImgSet
        seg_image = np.uint8(rgb.copy())
        # seg_image = np.uint8(rgb.copy())
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

        imgName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        cv2.imwrite(os.path.join(self.segResDir, imgName), mask * 255)
        cv2.imwrite(os.path.join(self.segVisDir, imgName), seg_image*mask[:,:,np.newaxis])

    def postProcess(self, idx, procImgSet, bb, img2bb, bb2img, kps, processed_kpts):

        rgbName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        depthName = str(self.camID) + '_' + format(idx, '04') + '.png'
        cv2.imwrite(os.path.join(self.rgbCropDir, rgbName), procImgSet[0])
        cv2.imwrite(os.path.join(self.depthCropDir, depthName), procImgSet[1])

        # vis = paint_kpts(None, procImgSet[0], processed_kpts)
        # imgName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        # cv2.imwrite(os.path.join(self.debug_vis, imgName), vis)

        meta_info = {'bb': bb, 'img2bb': np.float32(img2bb),
                     'bb2img': np.float32(bb2img), 'kpts': np.float32(kps), 'kpts_crop': np.float32(processed_kpts)}

        metaName = str(self.camID) + '_' + format(idx, '04') + '.pkl'
        jsonPath = os.path.join(self.metaDir, metaName)
        with open(jsonPath, 'wb') as f:
            pickle.dump(meta_info, f, pickle.HIGHEST_PROTOCOL)
        
        if self.info != None:
            with open(os.path.join(self.annotDir, "annot_%04d.json"), "w") as w:
                json.dump(self.info, w)

def preprocess_single_cam(db, tqdm_func, global_tqdm):
    with tqdm_func(total=len(db)) as progress:
        progress.set_description(f"{db.seq} - {db.trial} [{db.camID}]")
        mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
        for idx in range(len(db)):
            images = db.getItem(idx)
            images = db.undistort(images)
            output = db.procImg(images, mp_hand)
            if output[0] is None:
                continue
            else:
                bb, img2bb, bb2img, procImgSet, kps = output
            procKps = db.translateKpts(np.copy(kps), img2bb)
            # matting, mattedRgb = db.backgroundMatting(images, bgs, bb)
            # procImgSet.append(matting)
            # procImgSet.append(mattedRgb)
            db.postProcess(idx, procImgSet, bb, img2bb, bb2img, kps, procKps)
            if flag_segmentation:
                db.segmenation(idx, procImgSet, procKps)
            
            progress.update()
            global_tqdm.update()

    return True

def error_callback(result):
    print("Error!")

def done_callback(result):
    print("Done. Result: ", result)

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

    tasks = []
    process_count = 8
    total_count = 0

    for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
        seqDir = os.path.join(rootDir, seqName)
        print("---------------start preprocess seq : %s ---------------" % (seqName))
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            for camID in camIDset:
                db = loadDataset(FLAGS.db, seqName, trialName)
                db.init_cam(camID)
                db.init_info()
                total_count += len(db)
                tasks.append((preprocess_single_cam, (db,)))

        pool = TqdmMultiProcessPool(process_count)
        with tqdm.tqdm(total=total_count) as global_tqdm:
            # global_tqdm.set_description("total")
            pool.map(global_tqdm, tasks, error_callback, done_callback)

    print("---------------end preprocess seq : %s ---------------" % (seqName))
    print(time.ctime())

if __name__ == '__main__':
    app.run(main)