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


# others
import cv2
import time
import json
import pickle
import copy
import tqdm



### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230802', 'target db Name')   ## name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
flags.DEFINE_string('camID', 'mas', 'main target camera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)


### Config ###
baseDir = os.path.join(os.getcwd(), 'dataset')
handResultDir = os.path.join(baseDir, FLAGS.db) + '_hand'
camResultDir = os.path.join(baseDir, FLAGS.db) + '_cam'

## TODO
"""
    - Issues in camResultDir 
    - segmentation process(original) requires cam_mas_intrinsics.txt ... etc.
        - check HOnnotate_refine.HOdatasets.ho3d_multicamera.dataset.py
        - convert the process to utilize our format (230612_cameraInfo.txt)
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
        self.handDir = handResultDir

        self.rgbDir = os.path.join(self.dbDir, 'rgb')
        self.depthDir = os.path.join(self.dbDir, 'depth')

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

        self.rgbCropDir = None
        self.depthCropDir = None
        self.metaDir = None

        self.bbox_width = 640
        self.bbox_height = 480
        self.prev_bbox = [0, 0, 640, 480]

        intrinsics, dist_coeffs, extrinsics = LoadCameraParams(os.path.join(camResultDir, "cameraParams.json"))
        self.intrinsics = intrinsics
        self.distCoeffs = dist_coeffs


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
        
        self.debugDir = os.path.join(self.dbDir, 'debug')
        self.mp_hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=threshold) 
        self.K = self.intrinsics[camID]
        self.dist = self.distCoeffs[camID]

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

        idx_to_coordinates = None
        #### extract image bounding box ####
        image = cv2.flip(rgb, 1)
        # Convert the BGR image to RGB before processing.
        results = self.mp_hand.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        kps = np.empty((21, 3), dtype=np.float32)
        kps[:] = np.nan

        if results.multi_hand_landmarks:
            #TODO: handle multi hands
            hand_landmark = results.multi_hand_landmarks[0]
            idx_to_coordinates = {}

            wrist_px = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y,
                                                                                  image_cols, image_rows)
            wristDepth = depth[wrist_px[1], image_cols - wrist_px[0]]

            for idx_land, landmark in enumerate(hand_landmark.landmark):
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                                  image_cols, image_rows)
                if landmark_px:
                    # landmark_px has fliped x axis
                    orig_x = image_cols - landmark_px[0]
                    idx_to_coordinates[idx_land] = [orig_x, landmark_px[1]]

                    # 3d keypoints from depth image and mediapipe
                    if wristDepth > 0:
                        kps[idx_land, 0] = image_cols - landmark_px[0]
                        kps[idx_land, 1] = landmark_px[1]
                        kps[idx_land, 2] = landmark.z * image_cols + wristDepth

        # consider only one hand, if both hands are detected utilize idx_to_coord_1
        # if tracking fails, use the previous bbox
        idx_to_coord = idx_to_coordinates

        if idx_to_coord is None:
            bbox = self.prev_bbox
        else:
            bbox = self.extractBbox(idx_to_coord, image_rows, image_cols)

        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)

        # need to update if bbox is not fixed size
        depthCrop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        procImgSet = [rgbCrop, depthCrop]

        self.prev_bbox = copy.deepcopy(bbox)

        return bbox, img2bb_trans, bb2img_trans, procImgSet, kps

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
    
    def segmenation(self, camID, idx, procImgSet, kps):
        if np.any(np.isnan(kps)):
            return

        rgb, _ = procImgSet
        seg_image = np.uint8(rgb.copy())
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
                db.init_cam(camID)
                pbar = tqdm.tqdm(range(int(len(db) / 4)))
                for idx in pbar:
                    images = db.getItem(idx, camID=camID)

                    images = db.undistort(images, camID)

                    bb, img2bb, bb2img, procImgSet, kps = db.procImg(images)
                    procKps = db.translateKpts(np.copy(kps), img2bb)

                    db.postProcess(idx, procImgSet, bb, img2bb, bb2img, kps, procKps, camID=camID)
                    if flag_segmentation:
                        db.segmenation(camID, idx, procImgSet, procKps)
                    pbar.set_description("(%s in %s) : (cam %s, idx %s) in %s" % (seqIdx, lenDBTotal, camID, idx, seqName))
        print("---------------end preprocess---------------")

if __name__ == '__main__':
    app.run(main)