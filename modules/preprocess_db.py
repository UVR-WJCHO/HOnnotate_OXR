import os
import os.path
import math
import cv2
import mediapipe as mp
import numpy as np
import copy, time
from absl import flags
from absl import app
import json
import pickle
import copy
import tqdm
from utils.processing import augmentation_real


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


FLAGS = flags.FLAGS

flags.DEFINE_string('db', '230104', 'target db Name') # name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
# flags.DEFINE_string('camID', 'mas', 'target camIDera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']

baseDir = '/home/uvr-1080ti/projects/HOnnotate_OXR/dataset/'


class datasetRecord():
    def __init__(self, db, seq):
        self.seq = seq
        
        self.dbDir = os.path.join(baseDir, db, seq)
        self.handDir = os.path.join(baseDir, db) + '_hand'
        
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
            
           
        self.rgbCropDir = None
        self.depthCropDir = None
        self.metaDir = None
        
        self.bbox_width = 640
        self.bbox_height = 480
        self.prev_bbox = [0, 0, 640, 480]
        self.kps_all = None
        

    def __len__(self):
        return len(os.listdir(self.rgbDir))

    def setSavePath(self, camID):           
        self.rgbCropDir = os.path.join(self.dbDir, 'rgb_crop', camID)
        self.depthCropDir = os.path.join(self.dbDir, 'depth_crop', camID)
        self.metaDir = os.path.join(self.dbDir, 'meta', camID)
        self.debugDir = os.path.join(self.dbDir, 'debug')
    
    def loadKps(self):
        kpsPath = os.path.join(self.handDir, 'hand_result', self.seq, 'handDetection_uvd.json')
        
        assert os.path.exists(kpsPath), 'handDetection_uvd.json does not exist'
        
        with open(kpsPath, 'rb') as f:
            self.kps_all = json.load(f)


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
    
    
    def getKps(self, idx, camID='mas'):
        dict_key = str(camID) + '_' + str(camID)
        return np.asarray(self.kps_all[dict_key])[idx]     # (frames) * 21 * 3
    
    
    def procImg(self, images):
        rgb, depth = images

        image_rows, image_cols, _ = rgb.shape

        #### extract image bounding box ####
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
            image = cv2.flip(rgb, 1)
            # image = np.copy(img)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            idx_to_coord_0 = None
            idx_to_coord_1 = None

            hand_idx = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    idx_to_coordinates = {}
                    for idx_land, landmark in enumerate(hand_landmarks.landmark):
                        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                       image_cols, image_rows)
                        if landmark_px:
                            # landmark_px has fliped x axis
                            orig_x = image_cols - landmark_px[0]
                            idx_to_coordinates[idx_land] = [orig_x, landmark_px[1]]
                    if hand_idx == 0:
                        idx_to_coord_0 = idx_to_coordinates
                        hand_idx += 1
                    else:
                        idx_to_coord_1 = idx_to_coordinates
        
        # consider only one hand, if both hands are detected utilize idx_to_coord_1
        # if tracking fails, use the previous bbox
        if idx_to_coord_0 is None:
            bbox = self.prev_bbox
        else:
            bbox = self.extractBbox(idx_to_coord_0, image_rows, image_cols)
            
        rgbCrop, img2bb_trans, bb2img_trans, _, _,  = augmentation_real(rgb, bbox, flip=False)
        
        # need to update if bbox is not fixed size
        depthCrop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        
        procImgSet = [rgbCrop, depthCrop]
                
        
        self.prev_bbox = copy.deepcopy(bbox)
        
        return bbox, img2bb_trans, bb2img_trans, procImgSet
    

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
        
        
def main(argv):
    
    rootDir = os.path.join(baseDir, FLAGS.db)
    total_seq = len(os.listdir(rootDir))
    for seqIdx, seq in enumerate(sorted(os.listdir(rootDir))):
        d = os.path.join(rootDir, seq)
        
        
        if os.path.isdir(d):            
            db = datasetRecord(FLAGS.db, seq)
            db.loadKps()
            
            # db includes data for [mas, sub1, sub2, sub3]
            for camID in camIDset:
                pbar = tqdm.tqdm(range(int(len(db) / 4)))
                for idx in pbar:
                    db.setSavePath(camID)
                    
                    images = db.getItem(idx, camID=camID)
                    kps = db.getKps(idx, camID=camID)

                    bb, img2bb, bb2img, procImgSet = db.procImg(images)
                    procKps = db.translateKpts(kps, img2bb)
                    
                    rgbCrop = procImgSet[0]
                    # if not math.isnan(procKps[0, 0]):                
                    #     for joint_num in range(21):
                    #         cv2.circle(rgbCrop, center=(int(procKps[joint_num][0]), int(procKps[joint_num][1])), radius=3, color=[139, 53, 255], thickness=-1)

                    #     imgName = 'debug_' + format(idx, '04') + '.png'
                    #     cv2.imwrite(os.path.join(db.debugDir, imgName), rgbCrop)
                    
                    db.postProcess(idx, procImgSet, bb, img2bb, bb2img, kps, procKps, camID=camID)
                    pbar.set_description("(%s in %s) : (cam %s, idx %s) in %s" % (seqIdx, total_seq, camID, idx, seq))

            print("end")
    
    
if __name__ == '__main__':
    app.run(main)