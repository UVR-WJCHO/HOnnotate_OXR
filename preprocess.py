import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
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
from modules.deepLabV3plus.oxr_predict import predict as deepSegPredict

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
from natsort import natsorted


flag_check_vert_marker_pair = False

### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230912', 'target db Name')   ## name ,default, help
flags.DEFINE_string('cam_db', '230912_cam', 'target cam db Name')   ## name ,default, help
flags.DEFINE_float('mp_value', 0.88, 'target cam db Name')

flags.DEFINE_string('seq', None, 'target cam db Name')   ## name ,default, help
flags.DEFINE_integer('start', None, 'start idx of sequence(ordered)')
flags.DEFINE_integer('end', None, 'end idx of sequence(ordered)')

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

segIndices = [1,5,9,13,17,1]
palmIndices = [0,1,5,9,13,17,0]
thumbIndices = [0,1,2,3,4]
indexIndices = [5,6,7,8]
middleIndices = [9,10,11,12]
ringIndices = [13,14,15,16]
pinkyIndices = [17,18,19,20]
lineIndices = [palmIndices, thumbIndices, indexIndices, middleIndices, ringIndices, pinkyIndices]

### Manual Flags (remove after debug) ###
flag_preprocess = True
flag_segmentation = False
flag_deep_segmentation = True


num_global = 0

CFG_TIP_NAME = ['thumb', 'index', 'middle', 'ring', 'pinky']

### check config date ###
CFG_DATE = None
if FLAGS.db in ['230829', '230830', '230904', '230905', '230906', '230907', '230908']:
    CFG_DATE = '230829~230908'
elif FLAGS.db in ['230909', '230910', '230911', '230912', '230913']:
    CFG_DATE = '230909~230913'
elif FLAGS.db in ['230914']:
    CFG_DATE = '230914'
elif FLAGS.db in ['230915', '230916', '230917', '230918', '230919', '230920', '230921',  '230922', '230923', '230924', '230925',  '230926', '230927']:
    CFG_DATE = '230915~'
else:
    assert False, 'no CFG date matches, contact KAIST'


class deeplab_opts():
    def __init__(self, object_id):
        self.model = 'deeplabv3plus_mobilenet'
        self.output_stride = 16
        self.gpu_id = '0'
        self.ckpt = "./modules/deepLabV3plus/checkpoints/%02d_best_deeplabv3plus_mobilenet_oxr_os16.pth" % int(object_id)

        assert os.path.isfile(self.ckpt), "no ckpt files for object %02d" % int(object_id)
        print("...loading ", self.ckpt)
        self.checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))

def deepSegmentation(image_name, rgb, deepSegMaskDir, deepSegVisDir, opts):
    # print(type(rgb))
    assert isinstance(rgb, np.ndarray), "rgb type is not np.ndarray"
    mask, vis_mask = deepSegPredict(rgb, opts)
    maskName = image_name + '.png'
    visName = image_name + '.jpg'
    if mask is not None and vis_mask is not None:
        mask.save(os.path.join(deepSegMaskDir, maskName))
        vis_mask.save(os.path.join(deepSegVisDir, visName))
    else:
        print("no deep segmentation ckpt")


class loadDataset():
    def __init__(self, db, seq, trial):
        self.seq = seq # 230612_S01_obj_01_grasp_1
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

        self.segfgDir = os.path.join(self.dbDir, 'segmentation_fg')
        self.deepSegDir = os.path.join(self.dbDir, 'segmentation_deep')

        for camID in camIDset:
            os.makedirs(os.path.join(self.dbDir, 'rgb_crop', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'depth_crop', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'meta', camID), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'segmentation', camID, 'raw_seg_results'), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'segmentation', camID, 'visualization'), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'segmentation_deep', camID, 'raw_seg_results'), exist_ok=True)
            os.makedirs(os.path.join(self.dbDir, 'segmentation_deep', camID, 'visualization'), exist_ok=True)
            # os.makedirs(os.path.join(self.dbDir, 'masked_rgb', camID, 'bg'), exist_ok=True)

        os.makedirs(os.path.join(self.dbDir, 'visualizeMP'), exist_ok=True)
        self.debug_vis = os.path.join(self.dbDir, 'visualizeMP')
            
        self.rgbCropDir = None
        self.depthCropDir = None
        self.metaDir = None

        self.bbox_width = 640
        self.bbox_height = 480
        # 매뉴얼 하게 초기 bbox를 찾아서 설정해줌

        self.init_bbox = {}
        self.init_bbox["mas"] = [400, 60, 1120, 960]
        self.init_bbox["sub1"] = [360, 0, 1120, 960]
        self.init_bbox["sub2"] = [640, 180, 640, 720]
        self.init_bbox["sub3"] = [680, 180, 960, 720]


        self.prev_bbox = None
        self.wrist_px = None

        self.tip_db_dir = self.dbDir_result = os.path.join(baseDir, db+'_tip', seq, trial)

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
        self.marker_valid_idx_sampled = {}

        self.obj_db_Dir = os.path.join(baseDir, FLAGS.db+'_obj')

        obj_dir_name = "_".join(seq.split('_')[:-2]) # 230612_S01_obj_01
        self.obj_data_Dir = os.path.join(self.obj_db_Dir, obj_dir_name)

        self.obj_cam_ext = np.load(os.path.join(camResultDir, 'global_world.npy'))

        self.h = np.array([[0, 0, 0, 1]])
        self.obj_cam_ext = np.concatenate((self.obj_cam_ext, self.h), axis=0)

        # load 3mm marker pose
        obj_origin_name = '3mm.txt'
        if FLAGS.db+'_cam' != FLAGS.cam_db:
            obj_origin_name = '3mm_2.txt'
        if int(self.trial_num) == 0:
            print("... loading obj pose data for %s from %s "% (self.seq, obj_origin_name))

        obj_origin_data = os.path.join(self.obj_db_Dir, obj_origin_name)
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
        self.obj_pose_name = obj_dir_name + '_grasp_' + str("%02d"%int(self.grasp_id))+ '_' + str("%02d"%int(self.trial_num))
        obj_pose_data = os.path.join(self.obj_data_Dir, self.obj_pose_name+'.txt')
        # print(obj_pose_data)
        assert os.path.isfile(obj_pose_data),"no obj pose data : %s" % obj_pose_data

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

        ### check obj_pose is valid ###
        obj_pose_len = len(obj_data) - 1
        img_len = len(os.listdir(os.path.join(self.rgbDir, 'mas')))

        self.quit = False
        if obj_pose_len != img_len and abs(obj_pose_len-img_len) != 1:
            self.quit = True

        self.obj_data_all = obj_data
        self.marker_sampled['marker_num'] = self.obj_data_all['marker_num']

        self.marker_proj = np.zeros((self.marker_num, 2))

        # load object mesh info
        self.obj_class = self.obj_id + '_' + str(OBJType(int(self.obj_id)).name)
        self.obj_template_dir = os.path.join(baseDir, 'obj_scanned_models', self.obj_class)

        target_mesh_class = self.obj_class
        if int(self.obj_id) == 29:
            if int(self.grasp_id) == 12:
                target_mesh_class = '29_foldable_phone'
            else:
                target_mesh_class = '29_foldable_phone_2'

        # load object mesh data (new scanned object need to be load through pytorch3d 'load_obj'
        obj_mesh_path = os.path.join(self.obj_template_dir, target_mesh_class) + '.obj'


        # self.obj_mesh_data = self.read_obj(obj_mesh_path)
        self.obj_mesh_data = {}
        try:
            verts, faces, _ = load_obj(obj_mesh_path)
            self.obj_mesh_data['verts'] = verts
            self.obj_mesh_data['faces'] = faces.verts_idx
        except:
            print("no obj mesh data : %s" % obj_mesh_path)

        self.obj_scale = None


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

        # old
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

        self.prev_bbox = self.init_bbox[camID]

        self.tip_data_dir = os.path.join(self.tip_db_dir, camID)

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
        with open(os.path.join(os.getcwd(), 'annotation_template.json'), encoding='UTF8') as r:
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

        Ks = self.intrinsic_undistort[self.camID]
        info['calibration']['intrinsic'] = str(Ks[0,0]) + "," + str(Ks[1,1]) + "," + str(Ks[0,2]) + "," + str(Ks[1,2]) #fx, fy, cx, cy

        info['object']['id'] = self.obj_id
        info['object']['name'] = OBJType(int(self.obj_id)).name
        info['object']['marker_count'] = csv_dict['object.marker_count']
        info['object']['markers_data'] = csv_dict['object.markers_data']
        info['object']['pose_data'] = csv_dict['object.pose_data']
        self.info = info

    def updateObjdata(self, idx, save_idx, grasp_id):
        if str(idx) in self.obj_data_all:
            self.marker_sampled[str(save_idx)] = self.obj_data_all[str(idx)]

            marker_data = np.copy(self.obj_data_all[str(idx)])
            marker_data_cam, self.marker_proj = self.transform_marker_pose(marker_data)

            obj_pose_data, scale, marker_data_valid, valid_idx = self.fit_markerToObj(marker_data_cam, self.obj_class, self.obj_mesh_data, grasp_id)

            self.marker_cam_sampled[str(save_idx)] = marker_data_valid
            self.marker_valid_idx_sampled[str(save_idx)] = valid_idx
            self.obj_pose_sampled[str(save_idx)] = obj_pose_data
            if self.obj_scale == None:
                self.obj_scale = scale

        else:
            self.marker_sampled[str(save_idx)] = None
            self.marker_cam_sampled[str(save_idx)] = None
            self.obj_pose_sampled[str(save_idx)] = None
            self.marker_valid_idx_sampled[str(save_idx)] = None

    def transform_marker_pose(self, marker_poses_mocap):
        obj_cam_ext = self.obj_cam_ext
        # transform marker pose origin to master cam
        intr = self.intrinsics[self.camID]
        distC = self.distCoeffs[self.camID]

        coord_homo = np.concatenate((marker_poses_mocap.T, np.ones((1, self.marker_num))), axis=0)
        world_coord = obj_cam_ext @ coord_homo # world's coordinate
        world_coord = world_coord[:3].T

        ### debug ###
        projection = self.extrinsics[self.camID].reshape(3, 4)
        reprojected, _ = cv2.projectPoints(world_coord, projection[:, :3],
                                           projection[:, 3:], intr, distC)
        reprojected = np.squeeze(reprojected)

        # image = self.debug
        # for k in range(self.marker_num):
        #     point = reprojected[k, :]
        #     image = cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255))
        # cv2.imshow(f"debug marker to cam {self.camID}", image)
        # cv2.waitKey(0)

        return world_coord, reprojected

    def fit_markerToObj(self, marker_pose, obj_class, obj_mesh, grasp_id):
        ## except if marker pose is nan
        vertIDpermarker = CFG_vertspermarker[str(CFG_DATE)][str(obj_class)]

        ## exceptional cases
        if int(obj_class.split('_')[0]) == 29:
            if grasp_id == 12:
                vertIDpermarker = vertIDpermarker[0]
            else:
                vertIDpermarker = vertIDpermarker[1]

        pair_len = len(vertIDpermarker)

        target_marker_num = marker_pose.shape[0]
        valid_marker = []
        valid_idx = []
        for idx, i in enumerate(range(target_marker_num)):
            if idx == pair_len:
                break
            value = marker_pose[i, :]
            if not any(np.isnan(value)):
                valid_marker.append(value)
                valid_idx.append(idx)

        marker_pose = np.asarray(valid_marker)

        output_marker_pose = np.copy(marker_pose)
        # generate initial obj pose (4, 4)
        obj_init_pose = generate_pose([0,0,0],[0,0,0])

        obj_verts = obj_mesh['verts']
        verts_init = np.squeeze(np.array(obj_verts))
        verts_init = verts_init[vertIDpermarker, :] * 10.0

        verts_init = verts_init[valid_idx, :]

        # if obj_class in CFG_OBJECT_SCALE.keys():
        #     verts_init /= 10.0
        #     verts_init *= CFG_OBJECT_SCALE[obj_class]
        # scale factor 10, is .obj file has cm scale?

        # verts_pose = apply_transform(obj_init_pose, verts_init) * 100.0

        #verts_pose = torch.FloatTensor(verts_pose).unsqueeze(0)
        #marker_pose = torch.FloatTensor(marker_pose).unsqueeze(0)

        ## calculate scale of initial mesh
        scale = 1.0
        if obj_class in CFG_OBJECT_SCALE:
            mean = np.mean(verts_init, 0)
            k = verts_init.shape[0]
            expanded_mean = np.broadcast_to(mean, verts_init.shape)
            sub = verts_init - expanded_mean
            sub_squeeze = sub.flatten()
            s1 = np.sqrt(np.sum(np.square(sub_squeeze)) / k)

            ## calculate scale of marker mesh

            mean = np.mean(marker_pose, 0)
            k = marker_pose.shape[0]
            expanded_mean = np.broadcast_to(mean, marker_pose.shape)
            sub = marker_pose - expanded_mean
            sub_squeeze = sub.flatten()
            s2 = np.sqrt(np.sum(np.square(sub_squeeze)) / k)

            scale = s2 / s1

            verts_init = verts_init * scale

        if obj_class in CFG_OBJECT_SCALE_SPECIFIC.keys():
            verts_init = verts_init * CFG_OBJECT_SCALE_SPECIFIC[obj_class]

        if verts_init.shape[0] == 3 and obj_class in CFG_OBJECT_4th_POINT:
            newpoints_verts_init = np.expand_dims((verts_init[1] + verts_init[2] - verts_init[0]), axis=0)
            newpoints_marker_pose = np.expand_dims((marker_pose[1] + marker_pose[2] - marker_pose[0]), axis=0)
            verts_init = np.concatenate((verts_init, newpoints_verts_init), axis=0)
            marker_pose = np.concatenate((marker_pose, newpoints_marker_pose), axis=0)

        R, t = tf.solve_rigid_tf_np(verts_init, marker_pose)

        #R = R[0]
        #t = t[0]
        pose_calc = np.eye(4)
        pose_calc[:3, :3] = R[:3, :3]
        pose_calc[0, 3] = t[0]
        pose_calc[1, 3] = t[1]
        pose_calc[2, 3] = t[2]

        # verts_debug = np.squeeze(verts_pose)
        verts_debug = apply_transform(pose_calc, verts_init)# * 100.0
        marker_debug = np.squeeze(marker_pose)


        err = np.sum(abs(verts_debug - marker_debug), axis=1)
        err = np.average(err)

        ### debug
        if flag_check_vert_marker_pair:
            projection = self.extrinsics[self.camID].reshape(3, 4)
            marker_reproj, _ = cv2.projectPoints(marker_debug, projection[:, :3],
                                                 projection[:, 3:], self.intrinsics[self.camID],
                                                 self.distCoeffs[self.camID])
            marker_reproj = np.squeeze(marker_reproj)
            projection = self.extrinsics[self.camID].reshape(3, 4)
            vert_reproj, _ = cv2.projectPoints(verts_debug, projection[:, :3],
                                               projection[:, 3:], self.intrinsics[self.camID],
                                               self.distCoeffs[self.camID])
            vert_reproj = np.squeeze(vert_reproj)
            image = self.debug
            for k in range(marker_debug.shape[0]):
                point = marker_reproj[k, :]
                point_ = vert_reproj[k, :]
                image = cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255))
                image = cv2.circle(image, (int(point_[0]), int(point_[1])), 5, (0, 255, 0))
                cv2.imshow(f"debug marker to cam {self.camID}", image)
                #cv2.imwrite(f"/scratch/nia/HOnnotate_OXR/debug/{k}.png", image)
                cv2.waitKey(0)

            assert err < 22, f"wrong marker-vert fitting with err {err}, check obj in seq %s" % self.seq

        return pose_calc, scale, output_marker_pose, valid_idx

    def saveObjdata(self):
        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name+'_marker.pkl'), 'wb') as f:
            pickle.dump(self.marker_sampled, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name+'_marker_cam.pkl'), 'wb') as f:
            pickle.dump(self.marker_cam_sampled, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name + '_obj_pose.pkl'), 'wb') as f:
            pickle.dump(self.obj_pose_sampled, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name + '_obj_scale.pkl'), 'wb') as f:
            pickle.dump(self.obj_scale, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.obj_data_Dir, self.obj_pose_name + '_valid_idx.pkl'), 'wb') as f:
            pickle.dump(self.marker_valid_idx_sampled, f, pickle.HIGHEST_PROTOCOL)


    def getFgmask(self, idx):
        mask_fg_path = os.path.join(self.segfgDir, str(self.camID), str(self.camID) + "_%04d.png" % idx)
        mask_fg = cv2.imread(mask_fg_path)

        return mask_fg

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
        with open(os.path.join(self.annotCamDir, "anno_%04d.json"%save_idx), "w", encoding='cp949') as w:
            json.dump(self.info, w, ensure_ascii=False)

        self.debug = rgb.copy()
        ########## currently tip data processed on sampled data.
        ########## will be update to unsampled data
        tip_data_name = str(self.camID) + '_' + str(idx) + '.json'
        # tip_data_name = str(self.camID) + '_' + str(save_idx) + '.json'
        tip_data_path = os.path.join(self.tip_data_dir, tip_data_name)
        if CFG_exist_tip_db and os.path.exists(tip_data_path):
            with open(tip_data_path, "r") as data:
                tip_data = json.load(data)['annotations'][0]

            tip_kpts = {}
            for tip in tip_data:
                tip_name = tip['label']
                tip_2d = [tip['x'], tip['y']]
                tip_kpts[tip_name] = np.round(tip_2d, 2)

            self.tip_data = tip_kpts
        else:
            self.tip_data = None

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

        # crop initial image with each camera view's pre-defined bbox
        bbox = self.init_bbox[self.camID]
        rgb_init = np.copy(rgb)
        rgb_init = rgb_init[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        # run mediapipe
        results = mp_hand.process(cv2.cvtColor(rgb_init, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == 21:
            hand_landmark = results.multi_hand_landmarks[0]
            idx_to_coordinates = {}
            # wristDepth = depth[
            #     int(hand_landmark.landmark[0].y * image_rows), int(hand_landmark.landmark[0].x * image_cols)]
            for idx_land, landmark in enumerate(hand_landmark.landmark):
                landmark_px = [landmark.x * bbox[2] + bbox[0], landmark.y * bbox[3] + bbox[1]]
                if landmark_px:
                    # landmark_px has fliped x axis
                    idx_to_coordinates[idx_land] = [landmark_px[0], landmark_px[1]]

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

        if idx_to_coordinates is None:
            return None

        bbox = self.extractBbox(idx_to_coordinates, image_rows, image_cols)
        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)
        depthCrop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        procImgSet = [rgbCrop, depthCrop]
        self.prev_bbox = np.copy(bbox)

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

    def segmentation(self, idx, procImgSet, kps, img2bb):
        rgb, depth = procImgSet


        obj_kps = self.translateKpts(self.marker_proj, img2bb)
        ps = []
        for i in range(obj_kps.shape[0]):
            ps.append(np.asarray(obj_kps[i, :], dtype=int))

        hsv_image = np.uint8(rgb.copy())
        hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = 150
        rgb_filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("orig", rgb / 255)
        # cv2.imshow("result", result / 255)
        # cv2.waitKey(0)

        ### extract hand mask
        seg_image = np.uint8(rgb_filtered.copy())

        # apply fg
        # mask_fg_bin = np.ones(seg_image.shape[:2], np.uint8)
        # mask_fg_bin = np.where(mask_fg == 0, 0, mask_fg_bin)
        # seg_image = seg_image * mask_fg_bin[:, :, np.newaxis]
        # cv2.imshow("seg_image", seg_image / 255)
        # cv2.waitKey(0)

        # debug_image = seg_image.copy()
        mask = np.ones(seg_image.shape[:2], np.uint8) * 2
        # hand kpt for foreground
        for i in range(len(segIndices) - 1):
            point1 = np.int32(kps[0, :2])
            point2 = np.int32(kps[segIndices[i], :2])
            point3 = np.int32(kps[segIndices[i+1], :2])
            point4 = np.int32(kps[segIndices[i]+1, :2])
            cv2.line(mask, point1, point2, cv2.GC_FGD, 1)
            cv2.line(mask, point3, point2, cv2.GC_FGD, 1)
            cv2.line(mask, point4, point2, cv2.GC_FGD, 1)

            # cv2.line(debug_image, point1, point2, (255, 0, 0), 2)
            # cv2.line(debug_image, point3, point2, (255, 0, 0), 2)

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

        # extract obj mask
        # mask_obj = np.ones(seg_image.shape[:2], np.uint8) * 2
        # for p in ps:
        #     for p_ in ps:
        #         cv2.line(mask_obj, p, p_, cv2.GC_FGD, 2)
        # bgdModel = np.zeros((1, 65), np.float64)
        # fgdModel = np.zeros((1, 65), np.float64)
        # mask_obj, bgdModel, fgdModel = cv2.grabCut(seg_image, mask_obj, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        # mask_obj = np.where((mask_obj == 2) | (mask_obj == 0), 0, 1).astype('uint8')
        # mask_obj = np.where(depth > 1000, 0, mask_obj)
        #
        # mask_both = np.logical_and(mask_hand, mask_obj)
        # value_both = np.sum(mask_both)
        #
        # if self.camID == 'sub1':
        #     cv2.imshow("rgb", rgb / 255.0)
        #     cv2.imshow("mask_fg", mask_fg * 255.0)
        #     cv2.imshow("mask_hand", mask_hand * 255.0)
        #     # cv2.imshow("mask_obj", mask_obj * 255.0)
        #     # cv2.imshow("rgb_hand", np.uint8(rgb.copy()) * mask_hand[:, :, np.newaxis] / 255.0)
        #     cv2.waitKey(0)


        imgName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        cv2.imwrite(os.path.join(self.segResDir, imgName), mask_hand * 255)
        cv2.imwrite(os.path.join(self.segVisDir, imgName), rgb*mask_hand[:,:,np.newaxis])


    def postProcessNone(self, idx):
        meta_info = {'bb': None, 'img2bb': None,
                     'bb2img': None, 'kpts': None, 'kpts_crop': None, '2D_tip_gt':None, 'visibility': None}

        metaName = str(self.camID) + '_' + format(idx, '04') + '.pkl'
        jsonPath = os.path.join(self.metaDir, metaName)
        with open(jsonPath, 'wb') as f:
            pickle.dump(meta_info, f, pickle.HIGHEST_PROTOCOL)


    def postProcess(self, idx, procImgSet, bb, img2bb, bb2img, kps, processed_kpts, visibility):

        rgbName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        depthName = str(self.camID) + '_' + format(idx, '04') + '.png'
        cv2.imwrite(os.path.join(self.rgbCropDir, rgbName), procImgSet[0])
        cv2.imwrite(os.path.join(self.depthCropDir, depthName), procImgSet[1])
        # cv2.imwrite(os.path.join(self.rgbCropDir_result, rgbName), procImgSet[0])
        # cv2.imwrite(os.path.join(self.rgbCropDir_result, depthName), procImgSet[1])

        vis = paint_kpts(None, procImgSet[0], processed_kpts, visibility)
        imgName = str(self.camID) + '_' + format(idx, '04') + '.jpg'
        cv2.imwrite(os.path.join(self.debug_vis, imgName), vis)


        meta_info = {'bb': bb, 'img2bb': np.float32(img2bb),
                     'bb2img': np.float32(bb2img), 'kpts': np.float32(kps), 'kpts_crop': np.float32(processed_kpts),
                     '2D_tip_gt': self.tip_data, 'visibility': visibility}

        metaName = str(self.camID) + '_' + format(idx, '04') + '.pkl'
        jsonPath = os.path.join(self.metaDir, metaName)
        with open(jsonPath, 'wb') as f:
            pickle.dump(meta_info, f, pickle.HIGHEST_PROTOCOL)

    def computeVisibility(self, kpts):
        vis = np.ones(21)
        for i in range(21):
            kpt_x = kpts[i, 0]
            kpt_y = kpts[i, 1]
            kpt_z = kpts[i, 2]

            patch = [kpt_x-20, kpt_x+20, kpt_y-20, kpt_y+20]

            x_inrange = (kpts[:, 0] > patch[0]) & (kpts[:, 0] < patch[1])
            y_inrange = (kpts[:, 1] > patch[2]) & (kpts[:, 1] < patch[3])
            point_inrange = x_inrange & y_inrange
            idxlist_near = [idx for idx, x in enumerate(point_inrange) if x]
            idxlist_near.remove(i)
            if len(idxlist_near) != 0:
                for idx_near in idxlist_near:
                    if kpts[idx_near, 2] < kpt_z:
                        vis[i] = 0
            else:
                vis[i] = 1

        return vis



"""
def preprocess_single_cam(db, tqdm_func, global_tqdm):
    with tqdm_func(total=len(db)) as progress:
        progress.set_description(f"{db.seq} - {db.trial} [{db.camID}]")
        mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
        db.init_info()

        save_idx = 0
        for idx in range(len(db)):
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
                    db.segmentation(save_idx, procImgSet, procKps, img2bb)
                if flag_deep_segmentation:
                    db.deepSegmentation(save_idx, procImgSet)

            progress.update()
            global_tqdm.update()
            save_idx += 1

        db.saveObjdata()

    return True
"""
def preprocess_multi_cam(dbs, tqdm_func, global_tqdm):

    with tqdm_func(total=len(dbs[0])) as progress:
        progress.set_description(f"{dbs[0].seq} - {dbs[0].trial}")
        mp_hand_list = []
        for i in range(len(dbs)):
            mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.90, min_tracking_confidence=FLAGS.mp_value)
            mp_hand_list.append(mp_hand)

        for db in dbs:
            db.init_info()

        save_idx = 0
        for idx in range(len(dbs[0])):
            if idx % 3 != 0:
                progress.update()
                global_tqdm.update()
                continue

            images_list = []
            output_list = []
            for db, mp_hand in zip(dbs, mp_hand_list):
                images = db.getItem(idx, save_idx)
                images = db.undistort(images)
                output = db.procImg(images, mp_hand)
                images_list.append(images)
                output_list.append(output)

            for db, images, output in zip(dbs, images_list, output_list):
                if output is not None:
                    bb, img2bb, bb2img, procImgSet, kps = output
                    procKps = db.translateKpts(np.copy(kps), img2bb)
                    visibility = db.computeVisibility(procKps)
                    db.postProcess(save_idx, procImgSet, bb, img2bb, bb2img, kps, procKps, visibility)

                    # if flag_segmentation:
                    #     db.segmentation(save_idx, procImgSet, procKps, img2bb)
                else:
                    db.postProcessNone(save_idx)
            # object data only on master cam
            dbs[0].updateObjdata(idx, save_idx, int(dbs[0].grasp_id))

            progress.update()
            global_tqdm.update()
            save_idx += 1

        dbs[0].saveObjdata()

    return True



def error_callback(result):
    print("Error!")

def done_callback(result):
    # print("Done. Result: ", result)
    return

################# depth scale value need to be update #################
def main(argv):
    ### check
    assert FLAGS.db == FLAGS.cam_db[:6], "wrong db-cam_db pair. check name"
    assert os.path.exists(os.path.join(baseDir, FLAGS.db)), "no {YYMMDD} directory. check."
    assert os.path.exists(os.path.join(baseDir, FLAGS.cam_db)), "no{YYMMDD}_cam directory. check."
    assert os.path.exists(os.path.join(baseDir, FLAGS.db + '_obj')), "no {YYMMDD}_obj directory. check."
    assert os.path.exists(os.path.join(baseDir, 'obj_scanned_models')), "no dataset/obj_scanned_models directory. check."
    assert os.path.exists(
        os.path.join(os.getcwd(), 'modules/deepLabV3plus/checkpoints')), "no segmentation checkpoint folder. check."


    t0 = time.time()
    ### Setup ###
    rootDir = os.path.join(baseDir, FLAGS.db)

    ### Hand pose initialization(mediapipe) ###
    '''
    [TODO]
        - consider two-hand situation (currently assume single hand detection)
    '''

    print("---------------start preprocess seq ---------------")
    process_count = 4

    tasks = []
    total_count = 0
    t1 = time.time()

    obj_unvalid_trials = []

    seq_list = natsorted(os.listdir(rootDir))
    if FLAGS.start != None and FLAGS.end != None:
        seq_list = seq_list[FLAGS.start:FLAGS.end]

    for seqIdx, seqName in enumerate(seq_list):
        if FLAGS.seq is not None and seqName != FLAGS.seq:
            continue

        seqDir = os.path.join(rootDir, seqName)
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            dbs = []
            for camID in camIDset:
                db = loadDataset(FLAGS.db, seqName, trialName)
                db.init_cam(camID)
                dbs.append(db)
                # total_count += len(db)
                # tasks.append((preprocess_single_cam, (db,)))

            if dbs[0].quit:
                print("wrong object pose data, continue to next trial")
                obj_unvalid_trials.append(seqName + '_' + trialName)
            else:
                total_count += len(dbs[0])
                tasks.append((preprocess_multi_cam, (dbs,)))

    # tasks = tasks[:12]

    print("(fill in google sheets) unvalid trials with wrong object pose data : ", obj_unvalid_trials)


    pool = TqdmMultiProcessPool(process_count)
    with tqdm.tqdm(total=total_count) as global_tqdm:
        # global_tqdm.set_description(f"{seqName} - total : ")
        pool.map(global_tqdm, tasks, error_callback, done_callback)
    print("---------------end preprocess ---------------")


    proc_time = round((time.time() - t1) / 60., 2)
    print("total process time : %s min" % (str(proc_time)))

    print("start segmentation - deeplab_v3")
    if flag_deep_segmentation:
        seq_list = natsorted(os.listdir(rootDir))
        if FLAGS.start != None and FLAGS.end != None:
            seq_list = seq_list[FLAGS.start:FLAGS.end]

        for seqIdx, seqName in enumerate(seq_list):
            if FLAGS.seq is not None and seqName != FLAGS.seq:
                continue
            print("seq : %s" % seqName)
            seqDir = os.path.join(rootDir, seqName)
            for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):

                db = loadDataset(FLAGS.db, seqName, trialName)
                if db.quit:
                    continue

                opts = deeplab_opts(db.obj_id)

                dbDir = os.path.join(baseDir, FLAGS.db, seqName, trialName)
                for camID in camIDset:
                    rgbCropDir = os.path.join(dbDir, 'rgb_crop', camID)

                    deepSegDir = os.path.join(dbDir, 'segmentation_deep', camID)
                    deepSegMaskDir = os.path.join(deepSegDir, 'raw_seg_results')
                    deepSegVisDir = os.path.join(deepSegDir, 'visualization')

                    target = os.listdir(rgbCropDir)
                    target = natsorted(target)
                    for image in tqdm.tqdm(target):
                        rgbPath = os.path.join(rgbCropDir, image)
                        assert os.path.exists(rgbPath), f'{rgbPath} rgb image does not exist'
                        rgb = cv2.imread(rgbPath)
                        image_name = image[:-4]
                        deepSegmentation(image_name, rgb, deepSegMaskDir, deepSegVisDir, opts)

    print("end segmentation - deeplab_v3")

    # target_mp_num = 60
    seq_list = natsorted(os.listdir(rootDir))
    if FLAGS.start != None and FLAGS.end != None:
        seq_list = seq_list[FLAGS.start:FLAGS.end]

    for seqIdx, seqName in enumerate(seq_list):
        if FLAGS.seq is not None and seqName != FLAGS.seq:
            continue
        total_num = 0
        seqDir = os.path.join(rootDir, seqName)
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            # mp_num_list = []
            for camID in camIDset:
                anno_dir = os.path.join(baseDir, FLAGS.db+'_result', seqName, trialName, 'annotation', camID)
                num_anno = len(os.listdir(anno_dir))
                total_num += num_anno

            #     mp_dir = os.path.join(baseDir, FLAGS.db, seqName, trialName, 'rgb_crop', camID)
            #     num_mp = len(os.listdir(mp_dir))
            #     mp_num_list.append(num_mp)
            # mp_num_list.sort()

            # if num_anno < 60:
            #     target_mp_num = 50
            # else:
            #     target_mp_num = 60

            ## not for release
            # if mp_num_list[-2] < target_mp_num:
            #     print("[!] seq %s has not enough hand results, try with --seq {seq_name} --mp_value 0.85"%seqName)

        print("[LOG] total json # in seq %s : %s" % (seqName, total_num))

    print("[log] total processed time : ", round((time.time() - t0) / 60., 2))

    print("(fill in google sheets) unvalid trials with wrong object pose data : ", obj_unvalid_trials)

if __name__ == '__main__':

    app.run(main)
