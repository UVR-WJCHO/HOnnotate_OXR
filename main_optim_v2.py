import os
import sys

sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models/slim'))

# segmentation
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
import numpy as np

# optimization
from modules.utils.loadParameters import LoadCameraMatrix, LoadDistortionParam
from renderer_v2_pytorch import optimizer_torch

# others
import cv2
import time
import multiprocessing as mlp
import json
import pickle
import cop
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

rootDir = os.path.join(baseDir, FLAGS.db)
lenDBTotal = len(os.listdir(rootDir))

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


def handOptim(seqName, camParamList, metaList, imgList, flag_multi=False):

    intrinsics, extrinsics, distCoeffs = camParamList

    extrinsic_mas = np.copy(extrinsics['mas'])
    for camID in camIDset:
        # mano world is cm? scale translation value
        extrinsics[camID][:, -1] = np.copy(extrinsics[camID][:, -1]) / 10.0

        # extrinsics[camID][0, 0] = extrinsics[camID][0, 0] * -1.0
        # extrinsics[camID][1, 1] = extrinsics[camID][1, 1] * -1.0
        # extrinsics[camID][2, -1] = extrinsics[camID][2, -1] * -1.0

        # extrinsics[camID][0, :] = extrinsics[camID][0, :] * -1.0
        # extrinsics[camID][1, :] = extrinsics[camID][1, :] * -1.0
        # extrinsics[camID][:, 2] = extrinsics[camID][:, 2] * -1.0

    optm = optimizer_torch(camIDset, [intrinsics, extrinsics], flag_multi=flag_multi)

    # run each frame with set of metas/images
    for metas, imgs in zip(metaList, imgList):
        rgbSet = []
        depthSet = []
        segSet = []
        camSet = []
        for name in imgs:
            camID = name[:-5]
            rgb = os.path.join(rootDir, seqName, 'rgb_crop', camID, name+'.png')
            depth = os.path.join(rootDir, seqName, 'depth_crop', camID, name+'.png')
            seg = os.path.join(rootDir, seqName, 'segmentation', camID, 'raw_seg_results', name+'.png')

            rgb = np.asarray(cv2.imread(rgb))
            depth = np.asarray(cv2.imread(depth, cv2.IMREAD_UNCHANGED)).astype(float)
            seg = np.asarray(cv2.imread(seg, cv2.IMREAD_UNCHANGED))

            rgbSet.append(rgb)
            depthSet.append(depth)
            segSet.append(seg)

        for camID in camIDset:
            camSet.append([intrinsics[camID], extrinsics[camID]])

        data = [camSet, rgbSet, depthSet, segSet, metas]

        if not flag_multi:
            optm.run(data)
        else:
            optm.run_multiview(data)

    # dump optimized mano parameters as pkl
    return None


def getCam():
    with open(os.path.join(camResultDir, "cameraParamsBA.json")) as json_file:
        cameraParams = json.load(json_file)
        cameraParams = {camera: np.array(cameraParams[camera]) for camera in cameraParams}

    camInfoName = FLAGS.db + '_cameraInfo.txt'
    intrinsicMatrices = LoadCameraMatrix(os.path.join(camResultDir, camInfoName))
    distCoeffs = {}
    distCoeffs["mas"] = LoadDistortionParam(os.path.join(camResultDir, "mas_intrinsic.json"))
    distCoeffs["sub1"] = LoadDistortionParam(os.path.join(camResultDir, "sub1_intrinsic.json"))
    distCoeffs["sub2"] = LoadDistortionParam(os.path.join(camResultDir, "sub2_intrinsic.json"))
    distCoeffs["sub3"] = LoadDistortionParam(os.path.join(camResultDir, "sub3_intrinsic.json"))

    ### no translation in mano space ###
    for camID in camIDset:
        cameraParams[camID] = cameraParams[camID].reshape(3, 4)

    camParamList = intrinsicMatrices, cameraParams, distCoeffs
    return camParamList

def getMeta(files, seq):
    finalPklDataList = []
    ind = 0

    while (ind < len(files)):
        file = files[ind]
        resultsDir = os.path.join(baseDir, FLAGS.db)

        with open(os.path.join(resultsDir, seq, 'meta', file), 'rb') as f:
            pklData = pickle.load(f)
            finalPklDataList.append(pklData)
        ind = ind + 1

    return finalPklDataList


def main(argv):
    ### Setup ###

    ### Multi-view object pose optimization ###
    '''
    [TODO]
        - Load object pose & mesh (currently ICG)
        - render the scene (object only, utilize pytorch3D)
        - compute depth map error, silhouette error for each view
        - optimize object initial pose per frame (torch-based)
    '''
    if flag_MVobj:
        print("TODO")


    ### Multi-view hand-object pose optimization ###
    '''
    [Current]
        - optimization.NIA_handPoseMultiview.py
        - tensorflow_v1 based optimization (without depth rendering) 
        - only minimizing losses between 'mano mesh pose - multiview hand pose'
    
    [TODO]
        - torch based optimization
        - pytorch3D rendering (TW.H)
        - adopt losses from HOnnotate (optimization.handPoseMultiframe.py)    
    '''
    if flag_MVboth:
        # load camera param
        camParamList = getCam()     # intrinsicMatrices, cameraParams(R, T), distCoeffs

        # load meta data(hand results)
        for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
            pklDataperCam = []      # [[mas_pkl_list], [sub1_pkl_list], ...]
            imgNameperCam = []
            for camID in camIDset:
                pklFilesList = os.listdir(os.path.join(rootDir, seqName, 'meta', camID))
                frameList = [ff[:-4] for ff in pklFilesList if 'pkl' in ff]
                frameList = sorted(frameList)
                imgNameperCam.append(frameList)

                pklFilesList = [camID + '/' + f + '.pkl' for f in frameList]
                pklDataList = getMeta(pklFilesList, seqName)
                pklDataperCam.append(pklDataList)

            metas = []
            imgs = []
            frameLen = len(imgNameperCam[0])
            for i in range(frameLen):
                metasperFrame = []
                imgsperFrame = []
                for camIdx in range(len(camIDset)):
                    meta = pklDataperCam[camIdx][i]
                    img = imgNameperCam[camIdx][i]
                    metasperFrame.append(meta)
                    imgsperFrame.append(img)

                metas.append(metasperFrame)
                imgs.append(imgsperFrame)

            # 현재 sequence 별로 초기화 진행. 낭비. 이후에 db 별로 초기화하도록 변경.
            handOptim(seqName, camParamList, metas, imgs, flag_multi=True)

    ### Multi-frame pose refinement ###
    '''
    [TODO]
        - optimization over multi-frames while including every hand-object loss        
        - check optimization.handObjectRefinementMultiframe.py
    '''




    print("end")



if __name__ == '__main__':
    app.run(main)







