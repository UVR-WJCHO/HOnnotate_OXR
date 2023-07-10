import os
import sys

sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models/slim'))

# segmentation
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import models.deeplab.common as common
from utils.predictSegHandObject import getNetSess
from onlineAug.commonAug import networkData
from utils import inferenceUtils as infUti
from eval import evalSeg
from HOdatasets.commonDS import *
import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from absl import flags
from absl import app

# preprocess
import mediapipe as mp
from modules.utils.processing import augmentation_real
import numpy as np

# optimization
from modules.utils.loadParameters import LoadCameraMatrix, LoadDistortionParam
from renderer_pytorch import *

# others
import cv2
import time
import multiprocessing as mlp
import json
import pickle
import copy
import tqdm



### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230612', 'target db Name')   ## name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
flags.DEFINE_string('camID', 'mas', 'main target camera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)


### Config ###
baseDir = os.path.join(os.getcwd(), 'dataset')
handResultDir = os.path.join(baseDir, FLAGS.db) + '_hand'

## TODO
"""
    - Issues in camResultDir 
    - segmentation process(original) requires cam_mas_intrinsics.txt ... etc.
        - check HOnnotate_refine.HOdatasets.ho3d_multicamera.dataset.py
        - convert the process to utilize our format (230612_cameraInfo.txt)
"""
camResultDir = os.path.join(baseDir, FLAGS.db) + '_cam'


SEG_CKPT_DIR = 'HOnnotate_refine/checkpoints/Deeplab_seg'
YCB_MODELS_DIR = './HOnnotate_OXR/HOnnotate_refine/YCB_Models/models/'
YCB_OBJECT_CORNERS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../objCorners')
MANO_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../optimization/mano/models/MANO_RIGHT.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
dataset_mix = infUti.datasetMix.OXR_MULTICAMERA
numConsThreads = 1
w = 640
h = 480

### Manual Flags (remove after debug) ###
flag_preprocess = False
flag_segmentation = False
flag_MVobj = True
flag_MVboth = True


class loadDataset():
    def __init__(self, db, seq, flag_seg=False):
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
        kpsPath = os.path.join(self.handDir, self.seq, 'handDetection_uvd.json')
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
        return np.asarray(self.kps_all[dict_key])[idx]  # (frames) * 21 * 3

    def procImg(self, images):
        rgb, depth = images

        image_rows, image_cols, _ = rgb.shape

        #### extract image bounding box ####
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
            image = cv2.flip(rgb, 1)
            # image = np.copy(img)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

        # consider only one hand, if both hands are detected utilize idx_to_coord_1
        # if tracking fails, use the previous bbox
        idx_to_coord = idx_to_coordinates[0]
        if idx_to_coord is None:
            bbox = self.prev_bbox
        else:
            bbox = self.extractBbox(idx_to_coord, image_rows, image_cols)

        rgbCrop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(rgb, bbox, flip=False)

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

def runSegpostProc(dummy, consQueue, numImgs, numConsThreads):
    while True:
        queueElem = consQueue.get()
        predsDict = queueElem[0]
        ds = queueElem[1]
        jobID = queueElem[2]

        croppedImg = predsDict[common.IMAGE]
        if common.SEMANTIC in predsDict.keys():

            predsDict[common.SEMANTIC] = predsDict[common.SEMANTIC][0]

            assert len(ds.fileName.split('/')) == 3, 'Dont know this filename format'

            seq = ds.fileName.split('/')[0]
            camInd = ds.fileName.split('/')[1]
            id = ds.fileName.split('/')[2]

            finalSaveDir =  os.path.join(baseDir, FLAGS.db, seq, 'segmentation', str(camInd), 'visualization')
            finalRawSaveDir = os.path.join(baseDir, FLAGS.db, seq, 'segmentation', str(camInd), 'raw_seg_results')


            labelFullImg = np.zeros_like(ds.imgRaw)[:,:,0]
            patchSize = predsDict['bottomRight'] - predsDict['topLeft']

            scaleW = float(patchSize[0]) / float(w)
            scaleH = float(patchSize[1]) / float(h)
            labelPatch = cv2.resize(np.expand_dims(predsDict[common.SEMANTIC],2).astype(np.uint8),
                                    (int(ds.imgRaw.shape[1]*scaleW), int(ds.imgRaw.shape[0]*scaleH)),
                                    interpolation=cv2.INTER_NEAREST)
            labelFullImg[predsDict['topLeft'][1]:predsDict['bottomRight'][1], predsDict['topLeft'][0]:predsDict['bottomRight'][0]] = labelPatch

            # save predictions
            evalSeg.saveAnnotations(predsDict[common.SEMANTIC], croppedImg,
                                    finalSaveDir, id,
                                    raw_save_dir=finalRawSaveDir,
                                    also_save_raw_predictions=True, fullRawImg=labelFullImg)

        print('Frame %d of %d, (%s)' % (jobID, numImgs, ds.fileName))
        if jobID>=(numImgs-numConsThreads):
            return

def runSeg(fileListIn, numImgs, camID, seq):
    myG = tf.Graph()

    with myG.as_default():
        data = networkData(image=tf.placeholder(tf.uint8, shape=(h, w, 3)),
                           label=tf.placeholder(tf.uint8, shape=(h, w, 1)),
                           kps2D=None,
                           kps3D=None,
                           imageID='0',
                           h=h,
                           w=w,
                           outType=None,
                           dsName=None,
                           camMat=None)

    sess, g, predictions, dataPreProcDict = getNetSess(data, h, w, myG, SEG_CKPT_DIR)

    dsQueue, dsProcs = infUti.startInputQueueRunners(dataset_mix, splitType.TEST, FLAGS.db, seq, camID, numThreads=5, isRemoveBG=False, fileListIn=fileListIn)

    # start consumer threads
    consQueue = mlp.Queue(maxsize=100)
    procs = []
    for proc_index in range(numConsThreads):
        args = ([], consQueue, numImgs, numConsThreads)
        proc = mlp.Process(target=runSegpostProc, args=args)
        # proc.daemon = True

        proc.start()
        procs.append(proc)

    # start the network
    for i in range(numImgs):

        while(dsQueue.empty()):
            waitTime = 10*1e-3
            time.sleep(waitTime)

        ds = dsQueue.get()

        assert isinstance(ds, dataSample)

        startTime = time.time()
        predsDict = sess.run(predictions, feed_dict={data.image: ds.imgRaw},)
        # print('Runtime = %f'%(time.time() - startTime))

        labels = predsDict[common.SEMANTIC]
        labels[labels == 1] = 1
        labels[labels == 2] = 2
        labels[labels == 3] = 2
        predsDict[common.SEMANTIC] = labels

        consQueue.put([predsDict, ds, i])

    for proc in procs:
        proc.join()

    while(not consQueue.empty()):
        time.sleep(10*1e-3)

    consQueue.close()
    dsQueue.close()


################# depth scale value need to be update #################
def main(argv):
    ### Setup ###
    rootDir = os.path.join(baseDir, FLAGS.db)
    lenDBTotal = len(os.listdir(rootDir))




    ### Hand pose initialization(mediapipe) ###
    '''
    [Current]
        - preprocessed hand data(230612_hand) from mediapipe (modules.utils.detection.py)
        - we run mediapipe redundantly in preprocessing to get hand ROI (main.py loadDataset.procImg())
        - need to be merge.
    
    [TODO]
        - merge mediapipe part in single class
            - need to extract keypoint + bounding box + cropped image set
        - consider two-hand situation (currently assume single hand detection)
    '''




    ### Preprocess ###
    if flag_preprocess:
        print("---------------start preprocess---------------")
        for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
            db = loadDataset(FLAGS.db, seqName)
            db.loadKps()

            # db includes data for [mas, sub1, sub2, sub3]
            for camID in camIDset:
                db.setSavePath(camID)
                pbar = tqdm.tqdm(range(int(len(db) / 4)))
                for idx in pbar:
                    images = db.getItem(idx, camID=camID)
                    kps = db.getKps(idx, camID=camID)

                    bb, img2bb, bb2img, procImgSet = db.procImg(images)
                    procKps = db.translateKpts(kps, img2bb)
                    db.postProcess(idx, procImgSet, bb, img2bb, bb2img, kps, procKps, camID=camID)
                    pbar.set_description("(%s in %s) : (cam %s, idx %s) in %s" % (seqIdx, lenDBTotal, camID, idx, seqName))
        print("---------------end preprocess---------------")



    ### Segmentation ###
    if flag_segmentation:
        print("---------------start segmentation---------------")
        for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
            if not os.path.exists(os.path.join(baseDir, FLAGS.db, seqName, 'segmentation')):
                os.mkdir(os.path.join(baseDir, FLAGS.db, seqName, 'segmentation'))
                for camID in camIDset:
                    os.mkdir(os.path.join(baseDir, FLAGS.db, seqName, 'segmentation', camID))
                    os.mkdir(os.path.join(baseDir, FLAGS.db, seqName, 'segmentation', camID, 'visualization'))
                    os.mkdir(os.path.join(baseDir, FLAGS.db, seqName, 'segmentation', camID, 'raw_seg_results'))

            for camID in camIDset:
                fileListIn = os.listdir(os.path.join(baseDir, FLAGS.db, seqName, 'rgb_crop', camID))
                fileListIn = [os.path.join(seqName, camID, f[:-4]) for f in fileListIn if 'png' in f]
                fileListIn = sorted(fileListIn)

                numImgs = len(fileListIn)

                runSeg(fileListIn, numImgs, camID, seqName)
        print("---------------end segmentation---------------")




if __name__ == '__main__':
    app.run(main)







