import os
import sys

sys.path.insert(0,os.path.join(os.getcwd(), 'modules'))
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
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu


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
import shutil
from utils.dataUtils import *
# import modules.common.transforms as tf
from pytorch3d.io import load_obj
import modules.common.transforms as tf

flag_debug = False


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230905', 'target db Name')   ## name ,default, help
flags.DEFINE_string('cam_db', '230905_cam', 'target cam db Name')   ## name ,default, help
flags.DEFINE_string('obj_db', '230905_obj', 'target cam db Name')   ## name ,default, help
flags.DEFINE_string('obj_coord', '3', 'target cam coord idx in world_calib.py')   ## name ,default, help
flags.DEFINE_string('obj_cam', 'mas', 'target cam in world_calib.py')

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

image_cols, image_rows = 1080, 1920

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
numConsThreads = 1
w = 640
h = 480

segIndices = [1,5,9,13,17]
palmIndices = [0,1,5,9,13,17,0]
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
        self.db = db    # 230905
        assert len(seq.split('_')) == 6, 'invalid sequence name, ex. 230612_S01_obj_01_grasp_01'

        self.subject_id = seq.split('_')[1][1:]
        self.obj_id = seq.split('_')[3]
        self.grasp_id = seq.split('_')[5]
        self.trial = trial
        self.trial_num = trial.split('_')[1]

        ### create separate result dirs ###
        self.dbDir_result = os.path.join(baseDir, db+'_result', seq, trial)
        for camID in camIDset:
            os.makedirs(os.path.join(self.dbDir_result, 'rgb', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir_result, 'depth', camID), exist_ok=True)

        self.rgbDir_result = os.path.join(self.dbDir_result, 'rgb')
        self.depthDir_result = os.path.join(self.dbDir_result, 'depth')

        self.annotDir = os.path.join(self.dbDir_result, 'annotation')
        os.makedirs(self.annotDir, exist_ok=True)
        for camID in camIDset:
            os.makedirs(os.path.join(self.annotDir, camID), exist_ok=True)

        self.dbDir = os.path.join(baseDir, db, seq, trial)
        self.rgbDir = os.path.join(self.dbDir, 'rgb')
        self.depthDir = os.path.join(self.dbDir, 'depth')

        self.bgDir = os.path.join(baseDir, db) + '_bg'
        self.rgbBgDir = os.path.join(self.bgDir, 'rgb')
        self.depthBgDir = os.path.join(self.bgDir, 'depth')

        for camID in camIDset:
            os.makedirs(os.path.join(self.dbDir, 'rgb_crop', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'depth_crop', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'meta', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'segmentation', camID, 'raw_seg_results'), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'segmentation', camID, 'visualization'), exist_ok=True)
            # os.makedirs(os.path.join(self.dbDir, 'masked_rgb', camID, 'bg'), exist_ok=True)

        os.makedirs(os.path.join(self.dbDir, 'visualizeMP'), exist_ok=True)
        self.debug_vis = os.path.join(self.dbDir, 'visualizeMP')
            
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

        intrinsics, dist_coeffs, extrinsics, err = LoadCameraParams(os.path.join(camResultDir, "cameraParams.json"))
        self.intrinsics = intrinsics
        self.distCoeffs = dist_coeffs
        self.extrinsics = extrinsics
        self.calib_err = err

        # set save path for cameraInfo_undistort
        self.intrinsic_undistort_path = os.path.join(camResultDir, "cameraInfo_undistort.txt")
        self.intrinsic_undistort = {}
        for cam in CFG_CAMID_SET:
            new_camera, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics[cam], self.distCoeffs[cam], (image_rows, image_cols), 1, (image_rows, image_cols))
            self.intrinsic_undistort[cam] = new_camera

        if not os.path.isfile(self.intrinsic_undistort_path):
            print("creating undistorted intrinsic of each cam")
            with open(self.intrinsic_undistort_path, "w") as f:
                for cam in CFG_CAMID_SET:
                    new_camera = str(np.copy(self.intrinsic_undistort[cam]))
                    f.write(new_camera)
                    f.write("\n")

        ### object pose ###
        self.marker_sampled = {}
        self.marker_cam_sampled = {}
        self.obj_pose_sampled = {}

        self.obj_db_Dir = os.path.join(baseDir, FLAGS.obj_db)

        obj_dir_name = self.seq[:-9] # 230612_S01_obj_01
        self.obj_data_Dir = os.path.join(self.obj_db_Dir, obj_dir_name)

        self.obj_cam_ext = np.load(os.path.join(camResultDir, str(FLAGS.obj_coord) + '-world.npy'))
        self.obj_cam = str(FLAGS.obj_cam)

        self.h = np.array([[0, 0, 0, 1]])
        self.obj_cam_ext = np.concatenate((self.obj_cam_ext, self.h), axis=0)

        # load 3mm marker pose
        obj_origin_data = os.path.join(self.obj_db_Dir, '3mm.txt')
        assert os.path.isfile(obj_origin_data), "no 3mm pose data"
        with open(obj_origin_data, "r", encoding='euc-kr') as f:
            for i in range(5):
                _ = f.readline()
            line = f.readline().strip()
            line = line.split('\t')
            origin_x = float(line[1]) * 1000.       # 3mm.txt is m scale, other poses are mm scale
            origin_y = float(line[2]) * 1000.
            origin_z = float(line[3]) * 1000.

        # load marker set pose
        self.obj_pose_name = self.seq + '_0' + str(self.trial_num)
        obj_pose_data = os.path.join(self.obj_data_Dir, self.obj_pose_name+'.txt')
        assert os.path.isfile(obj_pose_data),"no obj pose data"

        obj_data = {}
        with open(obj_pose_data, "r") as f:
            line = f.readline().strip().split(' ')
            _ = f.readline()
            self.marker_num = int(float(line[0]))

            obj_data['marker_num'] = self.marker_num
            frame = 0
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.split(' ')
                line = [value for value in line if value != '']
                marker_pose = np.zeros((self.marker_num, 3))
                for i in range(self.marker_num):
                    marker_pose[i, 0] = float(line[i*3 + 1]) - origin_x
                    marker_pose[i, 1] = float(line[i*3 + 2]) - origin_y
                    marker_pose[i, 2] = float(line[i*3 + 3]) - origin_z
                obj_data[str(frame)] = marker_pose
                frame += 1

        self.obj_data_all = obj_data
        self.marker_sampled['marker_num'] = self.obj_data_all['marker_num']

        self.marker_proj = np.zeros((self.marker_num, 2))

        # load object mesh info
        obj_class = self.obj_id + '_' + str(OBJType(int(self.obj_id)).name)
        self.obj_template_dir = os.path.join(baseDir, 'obj_scanned_models', obj_class)

        # load object mesh data (new scanned object need to be load through pytorch3d 'load_obj'
        obj_mesh_path = os.path.join(self.obj_template_dir, obj_class) + '.obj'
        # self.obj_mesh_data = self.read_obj(obj_mesh_path)
        self.obj_mesh_data = {}
        self.obj_mesh_data['verts'], faces, _ = load_obj(obj_mesh_path)
        self.obj_mesh_data['faces'] = faces.verts_idx


    def __len__(self):
        return len(os.listdir(os.path.join(self.rgbDir, 'mas')))

    def init_cam(self, camID):
        #### calib err - 230825 ###
        #print("calib err:", self.calib_err)

        self.camID = camID
        self.rgbCropDir = os.path.join(self.dbDir, 'rgb_crop', camID)
        self.depthCropDir = os.path.join(self.dbDir, 'depth_crop', camID)

        # self.rgbCropDir_result = os.path.join(self.dbDir_result, 'rgb_crop', camID)
        # self.depthCropDir_result = os.path.join(self.dbDir_result, 'depth_crop', camID)

        self.metaDir = os.path.join(self.dbDir, 'meta', camID)
        segDir = os.path.join(self.dbDir, 'segmentation', camID)
        self.segVisDir = os.path.join(segDir, 'visualization')
        self.segResDir = os.path.join(segDir, 'raw_seg_results')

        #for background matting
        # self.maskedRgbDir = os.path.join(self.dbDir, 'masked_rgb', camID)
        # self.croppedBgDir = os.path.join(self.dbDir, 'masked_rgb', camID, 'bg')
        
        self.debugDir = os.path.join(self.dbDir, 'debug')
        self.K = self.intrinsics[camID]
        self.dist = self.distCoeffs[camID]
        self.new_camera = self.intrinsic_undistort[camID]

        self.prev_bbox = self.temp_bbox[camID]

    def init_info(self):
        self.annotCamDir = os.path.join(self.annotDir, self.camID)
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
        with open(os.path.join(os.getcwd(), 'annotation_template.json')) as r:
            info = json.load(r)
        #info keys 'info', 'actor', 'kinect_camera', 'infrared_camera', 'images', 'object', 'calibration', 'annotations', 'Mesh'
        # 기록해 놓은 actor 정보 받아오기
        info['info']['date_created'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
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

        info['infrared_camera'][0]['name'] = csv_dict['infrared_camera.name']
        info['infrared_camera'][0]['time'] = csv_dict['infrared_camera.time']
        info['infrared_camera'][0]['id'] = csv_dict['infrared_camera.id']
        # 고정값
        info['infrared_camera'][0]['height'] = "1080"
        info['infrared_camera'][0]['width'] = "1920"
        info['infrared_camera'][0]['frame'] = "60"
        info['infrared_camera'][0]['resolution'] = "0.14"
        info['infrared_camera'][0]['optics'] = "S"

        info['calibration']['error'] = self.calib_err
        Exts = self.extrinsics[self.camID]
        info['calibration']['extrinsic'] = Exts.tolist()
        Ks = self.intrinsics[self.camID]
        info['calibration']['intrinsic'] = str(Ks[0,0]) + "," + str(Ks[1,1]) + "," + str(Ks[0,2]) + "," + str(Ks[1,2]) #fx, fy, cx, cy

        info['object']['id'] = self.obj_id
        info['object']['name'] = OBJType(int(self.obj_id)).name
        info['object']['marker_count'] = csv_dict['object.marker_count']
        info['object']['markers_data'] = csv_dict['object.markers_data']
        info['object']['pose_data'] = csv_dict['object.pose_data']
        self.info = info

    def updateObjdata(self, idx, save_idx):
        if str(idx) in self.obj_data_all:
            self.marker_sampled[str(save_idx)] = self.obj_data_all[str(idx)]

            marker_data = np.copy(self.obj_data_all[str(idx)])
            marker_data_cam, self.marker_proj = self.transform_marker_pose(marker_data)
            self.marker_cam_sampled[str(save_idx)] = marker_data_cam

            obj_pose_data = self.fit_markerToObj(marker_data_cam, self.obj_id, self.obj_mesh_data)
            self.obj_pose_sampled[str(save_idx)] = obj_pose_data

        else:
            self.marker_sampled[str(save_idx)] = None
            self.marker_cam_sampled[str(save_idx)] = None
            self.obj_pose_sampled[str(save_idx)] = None

    def transform_marker_pose(self, marker_poses_mocap):
        obj_cam_ext = self.obj_cam_ext
        # transform marker pose origin to master cam
        extr = self.extrinsics[self.obj_cam]
        intr = self.intrinsics[self.camID]
        distC = self.distCoeffs[self.camID]

        coord_homo = np.concatenate((marker_poses_mocap.T, np.ones((1, self.marker_num))), axis=0)
        world_coord = obj_cam_ext @ coord_homo # master's coordinate
        projection = extr.reshape(3, 4)
        projection = np.concatenate((projection, self.h), axis=0)
        projection = np.linalg.inv(projection)
        world_coord = projection @ world_coord # camera's coordinate
        world_coord = world_coord[:3].T

        ### debug ###
        projection = self.extrinsics[self.camID].reshape(3, 4)
        reprojected, _ = cv2.projectPoints(world_coord, projection[:, :3],
                                           projection[:, 3:], intr, distC)
        reprojected = np.squeeze(reprojected)

        image = self.debug
        for k in range(4):
            point = reprojected[k, :]
            image = cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255))

            # cv2.imshow(f"debug marker to cam {self.camID}", image)
            # cv2.waitKey(0)

        return world_coord, reprojected

    def fit_markerToObj(self, marker_pose, obj_type, obj_mesh):

        vertIDpermarker = CFG_vertspermarker[str(OBJType(int(obj_type)).name)]
        obj_verts = obj_mesh['verts']
        # obj_faces = obj_mesh['faces']
        obj_verts_sample = obj_verts[vertIDpermarker, :]

        # generate initial obj pose (4, 4)
        obj_init_pose = generate_pose([0,0,0],[0,0,0])
        verts_pose = apply_transform(obj_init_pose, obj_verts_sample)

        marker_pose = torch.FloatTensor(marker_pose).unsqueeze(0)
        verts_pose = torch.FloatTensor(verts_pose).unsqueeze(0)

        R, t = tf.batch_solve_rigid_tf(verts_pose, marker_pose)

        R = R[0]
        t = t[0]

        pose_calc = np.eye(4)
        pose_calc[:3, :3] = R[:3, :3]
        pose_calc[0, 3] = t[0]
        pose_calc[1, 3] = t[1]
        pose_calc[2, 3] = t[2]

        verts_debug = np.squeeze(verts_pose.numpy())
        verts_debug = apply_transform(pose_calc, verts_debug)
        marker_debug = np.squeeze(marker_pose.numpy())

        err = np.sum(abs(verts_debug - marker_debug), axis=1)
        err = np.average(err)
        assert err < 100, f"wrong marker-vert fitting with err {err}, check vert idx"
        # verts_all = torch.FloatTensor(verts_all)
        # mesh = Meshes(verts=[verts_all], faces=[obj_faces]).to(self.device)

        return pose_calc


    def saveObjdata(self):
        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name+'_marker.pkl'), 'wb') as f:
            pickle.dump(self.marker_sampled, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name+'_marker_cam.pkl'), 'wb') as f:
            pickle.dump(self.marker_cam_sampled, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name + '_obj_pose.pkl'), 'wb') as f:
            pickle.dump(self.obj_pose_sampled, f, pickle.HIGHEST_PROTOCOL)

    def getItem(self, idx, save_idx):
        # camID : mas, sub1, sub2, sub3
        rgbName = str(self.camID) + '/' + str(self.camID) + '_' + str(idx) + '.jpg'
        depthName = str(self.camID) + '/' + str(self.camID) + '_' + str(idx) + '.png'

        rgbPath = os.path.join(self.rgbDir, rgbName)
        depthPath = os.path.join(self.depthDir, depthName)

        assert os.path.exists(rgbPath), f'{rgbPath} rgb image does not exist'
        assert os.path.exists(depthPath), f'{depthPath} depth image does not exist'

        rgb = cv2.imread(rgbPath)
        depth = cv2.imread(depthPath, cv2.IMREAD_ANYDEPTH)

        ### move sampled img
        rgbName_sampled = str(self.camID) + '/' + str(self.camID) + '_' + str(save_idx) + '.jpg'
        depthName_sampled = str(self.camID) + '/' + str(self.camID) + '_' + str(save_idx) + '.png'
        rgbPath_sampled = os.path.join(self.rgbDir_result, rgbName_sampled)
        depthPath_sampled = os.path.join(self.depthDir_result, depthName_sampled)
        cv2.imwrite(rgbPath_sampled, rgb)
        cv2.imwrite(depthPath_sampled, depth)


        self.info['images']['file_name'] = [os.path.join("rgb", str(self.camID) + '/' + str(self.camID) + '_' + str(save_idx) + '.jpg'), os.path.join("depth", str(self.camID) + '/' + str(self.camID) + '_' + str(save_idx) + '.png')]
        self.info['images']['id'] = str(self.db+'_result') + str(self.subject_id) + str(self.obj_id) + str(self.grasp_id) + str(self.trial_num) + str(camIDset.index(self.camID)) + str(idx)
        self.info['images']['width'] = rgb.shape[1]
        self.info['images']['height'] = rgb.shape[0]
        self.info['images']['date_created'] = dt.datetime.fromtimestamp(os.path.getctime(rgbPath)).strftime('%Y-%m-%d %H:%M')
        self.info['images']['frame_num'] = save_idx
        with open(os.path.join(self.annotCamDir, "anno_%04d.json"%save_idx), "w") as w:
            json.dump(self.info, w, ensure_ascii=False)


        self.debug = rgb.copy()

        return (rgb, depth)

    def undistort(self, images):
        rgb, depth = images
        rgb = cv2.undistort(rgb, self.K, self.dist, None, self.new_camera)
        depth = cv2.undistort(depth, self.K, self.dist, None, self.new_camera)

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

        if idx_to_coord is None:
            return [None]

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

    def segmenation(self, idx, procImgSet, kps, img2bb):
        rgb, depth = procImgSet

        procKps = self.translateKpts(self.marker_proj, img2bb)
        ps = []
        for i in range(procKps.shape[0]):
            ps.append(np.asarray(procKps[i, :], dtype=int))

        hsv_image = np.uint8(rgb.copy())
        hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 150
        rgb_filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("orig", rgb / 255)
        # cv2.imshow("result", result / 255)
        # cv2.waitKey(0)

        ### extract hand mask
        seg_image = np.uint8(rgb_filtered.copy())
        # debug_image = seg_image.copy()
        mask = np.ones(seg_image.shape[:2], np.uint8) * 2
        # hand kpt for foreground
        for i in range(len(segIndices)):
            point1 = np.int32(kps[0, :2])
            point2 = np.int32(kps[segIndices[i], :2])

            cv2.line(mask, point1, point2, cv2.GC_FGD, 1)
            # cv2.line(debug_image, point1, point2, (255, 0, 0), 2)

        # object marker for background
        for p in ps:
            for p_ in ps:
                cv2.line(mask, p, p_, cv2.GC_BGD, 2)
                # cv2.line(debug_image, p, p_, (0, 255, 0), 2)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(seg_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask_hand = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        mask_hand = np.where(depth > 1000, 0, mask_hand)

        # cv2.imshow("rgb", rgb / 255.0)
        # cv2.imshow("mask_hand", mask_hand * 255.0)
        # cv2.imshow("rgb_hand", np.uint8(rgb.copy()) * mask_hand[:, :, np.newaxis] / 255.0)
        # cv2.imshow("debug_image", debug_image / 255.0)
        # cv2.waitKey(0)


        imgName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        cv2.imwrite(os.path.join(self.segResDir, imgName), mask_hand * 255)
        cv2.imwrite(os.path.join(self.segVisDir, imgName), rgb*mask_hand[:,:,np.newaxis])

    def postProcess(self, idx, procImgSet, bb, img2bb, bb2img, kps, processed_kpts):

        rgbName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        depthName = str(self.camID) + '_' + format(idx, '04') + '.png'
        cv2.imwrite(os.path.join(self.rgbCropDir, rgbName), procImgSet[0])
        cv2.imwrite(os.path.join(self.depthCropDir, depthName), procImgSet[1])
        # cv2.imwrite(os.path.join(self.rgbCropDir_result, rgbName), procImgSet[0])
        # cv2.imwrite(os.path.join(self.rgbCropDir_result, depthName), procImgSet[1])

        vis = paint_kpts(None, procImgSet[0], processed_kpts)
        imgName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        cv2.imwrite(os.path.join(self.debug_vis, imgName), vis)

        meta_info = {'bb': bb, 'img2bb': np.float32(img2bb),
                     'bb2img': np.float32(bb2img), 'kpts': np.float32(kps), 'kpts_crop': np.float32(processed_kpts)}

        metaName = str(self.camID) + '_' + format(idx, '04') + '.pkl'
        jsonPath = os.path.join(self.metaDir, metaName)
        with open(jsonPath, 'wb') as f:
            pickle.dump(meta_info, f, pickle.HIGHEST_PROTOCOL)
        

def preprocess_single_cam(db, tqdm_func, global_tqdm):
    with tqdm_func(total=len(db)) as progress:
        progress.set_description(f"{db.seq} - {db.trial} [{db.camID}]")
        mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
        db.init_info()

        save_idx = 0
        for idx in range(len(db)):
            if idx < 5:
                continue
            if idx % 3 != 0:
                progress.update()
                global_tqdm.update()
                continue
            images = db.getItem(idx, save_idx)
            images = db.undistort(images)
            output = db.procImg(images, mp_hand)

            if not output[0] is None:
                bb, img2bb, bb2img, procImgSet, kps = output
                procKps = db.translateKpts(np.copy(kps), img2bb)
                db.postProcess(save_idx, procImgSet, bb, img2bb, bb2img, kps, procKps)

                db.updateObjdata(idx, save_idx)

                if flag_segmentation:
                    db.segmenation(save_idx, procImgSet, procKps, img2bb)

            progress.update()
            global_tqdm.update()
            save_idx += 1

        db.saveObjdata()

    return True

def error_callback(result):
    print("Error!")

def done_callback(result):
    # print("Done. Result: ", result)
    return

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
    process_count = 1
    total_count = 0

    for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
        seqDir = os.path.join(rootDir, seqName)
        print("---------------start preprocess seq : %s ---------------" % (seqName))
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            for camID in camIDset:
                db = loadDataset(FLAGS.db, seqName, trialName)
                db.init_cam(camID)
                total_count += len(db)
                tasks.append((preprocess_single_cam, (db,)))

        pool = TqdmMultiProcessPool(process_count)
        with tqdm.tqdm(total=total_count) as global_tqdm:
            global_tqdm.set_description(f"{seqName} - total : ")
            pool.map(global_tqdm, tasks, error_callback, done_callback)

        print("---------------end preprocess seq : %s ---------------" % (seqName))
    print(time.ctime())

if __name__ == '__main__':
    app.run(main)
