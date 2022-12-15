import os
import os.path
import cv2
import mediapipe as mp
import copy, time
from absl import flags
from utils.processing import augmentation_real

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


FLAGS = flags.FLAGS
flags.DEFINE_string('db', '221215_sample', 'target db Name') # name ,default, help
flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')


baseDir = '/home/uvr-1080ti/projects/HOnnotate_OXR/dataset/'


class datasetRecord():
    def __init__(self):

        self.dbDir = join(baseDir, FLAGS.db, FLAGS.seq)
        
        self.rgbDir = join(self.dbDir, 'rgb')
        self.depthDir = join(self.dbDir, 'depth')
        
        self.handDir = join(baseDir, FLAGS.db, '_hand')
        
        if not os.path.exists(os.path.join(self.dbDir, 'rgb_crop')):
            os.mkdir(os.path.join(self.dbDir, 'rgb_crop'))
        if not os.path.exists(os.path.join(self.dbDir, 'depth_crop')):
            os.mkdir(os.path.join(self.dbDir, 'depth_crop'))
        if not os.path.exists(os.path.join(self.dbDir, 'bbox')):
            os.mkdir(os.path.join(self.dbDir, 'bbox'))
            
        self.rgbCropDir = join(self.dbDir, 'rgb_crop')
        self.depthCropDir = join(self.dbDir, 'depth_crop')
        self.bboxDir = join(self.dbDir, 'bbox')

        
        self.bbox_width = 640
        self.bbox_height = 480
        self.prev_bbox = [0, 0, 640, 480]
        self.kps_all = None
        

    def __len__(self):
        return len(os.listdir(self.rgbDir))

    def loadKps(self):
        kpsPath = os.path.join(self.handDir, 'hand_results', FLAGS.seq + 'handDetection_uvd.json')
        
        assert os.path.exists(kpsPath), 'handDetection_uvd.json does not exist'
        
        self.kps_all = json.load(f)


    def getItem(self, idx, cam='mas'):
        # cam : mas, sub1, sub2, sub3
        imgName = str(cam) + '_' + str(idx) + '.png'
        
        rgbPath = os.path.join(self.rgbDir, imgName)
        depthPath = os.path.join(self.depthDir, imgName)
        
        assert os.path.exists(rgbPath), 'rgb image does not exist'
        assert os.path.exists(depthPath), 'depth image does not exist'

        rgb = cv2.imread(rgbPath)
        depth = cv2.imread(depthPath, cv2.IMREAD_ANYDEPTH)

        return zip(rgb, depth)
    
    def getKps(self, idx, cam='mas'):
        dict_key = str(cam) + '_' + str(cam))
        return self.kps_all[dict_key][idx]     # (frames) * 21 * 3
    
    
    def procImg(self, images):
        rgb, depth = images

        image_rows, image_cols, _ = rgb.shape

        #### extract image bounding box ####
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
            image = cv2.flip(rgb, 1)
            # image = np.copy(img)
            # Convert the BGR image to RGB before processing.
            t1 = time.time()
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("t : ", time.time() - t1)

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
        depthCrop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])
        
        procImgSet = zip(rgbCrop, depthCrop)
        
        return bb, img2bb, bb2img, procImgSet
    

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
    
    def postProcess(self, processed_images, bb, img2bb, bb2img, processed_kpts):
        
        # img_name = "./samples/221115_bowl_18_00/crop/mas_{}.png".format(idx)
        # cv2.imwrite(img_name, img_crop)

        # img_name = "./samples/221115_bowl_18_00/crop_depth/mas_{}.png".format(idx)
        # cv2.imwrite(img_name, depth_crop)
        assert NotImplementedError

if __name__ == '__main__':
    db = datasetRecord()
    db.loadKps()
    
    for idx in range(len(db)):
        # choose which camera to use [mas, sub1, sub2, sub3]
        images = db.getItem(idx, cam='mas')
        kps = db.getKps(idx, cam='mas')

        bb, img2bb, bb2img, procImgSet = db.procImg(images)
        procKps = db.translateKpts(kps, img2bb)

        db.postProcess(procImgSet, bb, img2bb, bb2img, procKps)







