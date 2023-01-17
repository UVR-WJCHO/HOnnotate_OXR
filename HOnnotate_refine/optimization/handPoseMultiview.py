import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import json
import pickle
from handUtils.lift2DJoints import lift2Dto3D, lift2Dto3DMultiFrame, lift2Dto3DMultiview
import utils.inferenceUtils as infUti
import multiprocessing as mlp
import handUtils.manoHandVis as manoVis
from HOdatasets.mypaths import *

from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera, datasetOXRMultiCamera
from HOdatasets.commonDS import *
from eval import utilsEval
import copy

from absl import flags
from absl import app

sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../modules/utils'))
from loadParameters import LoadCameraMatrix, LoadDistortionParam

FLAGS = flags.FLAGS

flags.DEFINE_string('db', '230104', 'target db Name') # name ,default, help
flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
# flags.DEFINE_string('camID', '0', 'Cam ID')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']

dataset_mix = infUti.datasetMix.OXR_MULTICAMERA

configDir = 'CPMHand'
baseDir = OXR_MULTI_CAMERA_DIR

beta = np.zeros((10,), dtype=np.float32)



"""
def incrementalFarthestSearch(objPklDataList, k):
    def distance(p, s):
        return np.linalg.norm(p-s)

    points = [pklData['rot'][0]/np.linalg.norm(pklData['rot'][0]) for pklData in objPklDataList]

    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(\
                                             np.random.randint(0, len(remaining_points) - 1)))
    for _ in range(k-1):
        distances = [distance(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, s))
        solution_set.append(remaining_points.pop(distances.index(max(distances))))
    solution_set_ind = []
    for p in solution_set:
        solution_set_ind.append(np.argmin(np.linalg.norm(np.stack(points, axis=0) - p, axis=1)))
    return solution_set, solution_set_ind

def selectCandidateFramesFarthest(files, numFrames, dataSet, frameSpacing=10, startFrame=0):
    '''
    Chooses the frames in the sequence for hand pose estimation
    criteria - the rotation vectors of the obj pose are placed as far apart as possible
    :param files: list of frame ids
    :param numFrames: number of frames to select
    :param frameSpacing: initial frame gap
    :param startFrame: staring frame number (not ID)
    :return: select frame IDs list and pklData list (contains 2D kps and conf scores)
    '''

    confThresh = 0.5
    confCntThresh = 12 # number of KPs where the conf is more than the 'confThresh'
    neighborThresh = 1 # number of frames around the frame where we look for good frames if the curr frame is bad
    objSegThresh = 0.010

    finalIndList = []
    handPklDataList = []
    ObjPklDataList = []
    ind = startFrame

    while(ind < len(files)):
        file = files[ind]
        resultsDirHand = join(baseDir, FLAGS.seq, configDir, 'Results_hand', FLAGS.camID)
        resultsDirObj = join(baseDir, FLAGS.seq, 'dirt_obj_pose', FLAGS.camID)



        with open(join(resultsDirObj, file.split('/')[-1]+'.pkl'), 'rb') as f:
            pklDataObj = pickle.load(f)
        with open(join(resultsDirHand, file.split('/')[-1]+'.pickle'), 'rb') as f:
            pklDataHand = pickle.load(f)

        _, ds = dataSet.createTFExample(itemType='hand', fileIn=file)
        # read obj segmentation
        objSeg = ds.segRaw == 1
        objOccupRatio = float(np.sum(objSeg))/float(objSeg.shape[0]*objSeg.shape[1])


        if (np.sum(pklDataHand['conf']>confThresh) > confCntThresh) and (objOccupRatio > objSegThresh):# and (int(file.split('/')[-1])%2 == 0):
            finalIndList.append(ind)
            handPklDataList.append(pklDataHand)
            ObjPklDataList.append(pklDataObj)
        else:
            # search within the neighbor
            isCandFound = False
            for i in range(-neighborThresh,neighborThresh,1):
                nInd = ind + i
                if (nInd < 0) or (nInd>=len(files)):
                    continue
                with open(join(resultsDirHand, files[nInd].split('/')[-1]+'.pickle'), 'rb') as f:
                    pklDataHand = pickle.load(f)
                with open(join(resultsDirObj, files[nInd].split('/')[-1]+'.pkl'), 'rb') as f:
                    pklDataObj = pickle.load(f)
                if (np.sum(pklDataHand['conf'] > confThresh) > confCntThresh) and (objOccupRatio > objSegThresh):# and (int(files[nInd].split('/')[-1])%2 == 0):
                    finalIndList.append(ind)
                    handPklDataList.append(pklDataHand)
                    ObjPklDataList.append(pklDataObj)
                    isCandFound = True
                    ind = nInd
                    break

        ind = ind + frameSpacing

    if len(finalIndList) >= numFrames:
        _, finalIndList = incrementalFarthestSearch(ObjPklDataList, numFrames)
        finalHandPklDataList = [handPklDataList[ind] for ind in finalIndList]
    else:
        raise Exception('Unable to find good candidate frames for hand pose initialization')

    print('Found %d/%d candidates with %d spacing...\n'%(len(finalIndList), numFrames, frameSpacing))

    return finalIndList, finalHandPklDataList

def selectCandidateFrames(files, numFrames, dataSet, frameSpacing=10, startFrame=0):
    '''
    Chooses the frames in the sequence for hand pose estimation
    criteria - equally (almost) and far placed + 2D joint estimation has good confidence + good object segmentation
    :param files: list of frame ids
    :param numFrames: number of frames to select
    :param frameSpacing: initial frame gap
    :param startFrame: staring frame number (not ID)
    :return: select frame IDs list and pklData list (contains 2D kps and conf scores)
    '''

    confThresh = 0.5
    confCntThresh = 12 # number of KPs where the conf is more than the 'confThresh'
    neighborThresh = 10 # number of frames around the frame where we look for good frames if the curr frame is bad
    objSegThresh = 0.010

    finalIndList = []
    finalPklDataList = []
    ind = startFrame

    while(ind < len(files)):
        file = files[ind]
        resultsDir = join(baseDir, FLAGS.seq, configDir, 'Results_hand', FLAGS.camID)



        with open(join(resultsDir, file.split('/')[-1]+'.pickle'), 'rb') as f:
            pklData = pickle.load(f)

        _, ds = dataSet.createTFExample(itemType='hand', fileIn=file)
        # read obj segmentation
        objSeg = ds.segRaw == 1
        objOccupRatio = float(np.sum(objSeg))/float(objSeg.shape[0]*objSeg.shape[1])


        if (np.sum(pklData['conf']>confThresh) > confCntThresh) and (objOccupRatio > objSegThresh):# and (int(file.split('/')[-1])%2 == 0):
            finalIndList.append(ind)
            finalPklDataList.append(pklData)
        else:
            # search within the neighbor
            isCandFound = False
            for i in range(-neighborThresh,neighborThresh,1):
                nInd = ind + i
                if (nInd < 0) or (nInd>=len(files)):
                    continue
                with open(join(resultsDir, files[nInd].split('/')[-1]+'.pickle'), 'rb') as f:
                    pklData = pickle.load(f)
                if (np.sum(pklData['conf'] > confThresh) > confCntThresh) and (objOccupRatio > objSegThresh) and (int(files[nInd].split('/')[-1])%2 == 0):
                    finalIndList.append(ind)
                    finalPklDataList.append(pklData)
                    isCandFound = True
                    ind = nInd
                    break

        ind = ind + frameSpacing

    if len(finalIndList) > numFrames:
        finalIndList = finalIndList[:numFrames]
        finalPklDataList = finalPklDataList[:numFrames]

    print('Found %d/%d candidates with %d spacing...\n'%(len(finalIndList), numFrames, frameSpacing))

    return finalIndList, finalPklDataList
"""

def getMeta(files):
    finalPklDataList = []
    ind = 0
    
    while(ind < len(files)):
        file = files[ind]
        resultsDir = join(baseDir, FLAGS.db)

        with open(join(resultsDir, FLAGS.seq, 'meta', file +'.pkl'), 'rb') as f:
            pklData = pickle.load(f)
            finalPklDataList.append(pklData)
        ind = ind + 1
        
    return finalPklDataList


def getFramewisePose(dummy, kpsDictList, camMat, beta, dataSet, saveCandImgDir):

    for kpsDict in kpsDictList:
        if os.path.exists(join(saveCandImgDir, kpsDict['imgID'].split('/')[-1]+'.pickle')):
            continue


        _, ds = dataSet.createTFExample(itemType='hand', fileIn=kpsDict['imgID'])
        imgRaw = ds.imgRaw
        kps3D, poseCoeff, beta, trans, err, _ = lift2Dto3D(kpsDict['KPS2D'], camMat,
                                                           kpsDict['imgID'].split('/')[-1], imgRaw,
                                                           weights=np.ones((21, 1), dtype=np.float32),
                                                           beta=beta,
                                                           rel3DCoordNormGT=None,#fcOut,
                                                           img2DGT=None,
                                                           outDir=saveCandImgDir
                                                          )

        newDict = {'KPS2D': kpsDict['KPS2D'],
                   'conf': kpsDict['conf'],
                   'imgID': kpsDict['imgID'],
                   'KPS3D': kps3D,
                   'poseCoeff': poseCoeff,
                   'beta': beta,
                   'trans': trans, 'err': err}

        with open(join(saveCandImgDir, kpsDict['imgID'].split('/')[-1]+'.pickle'), 'wb') as f:
            pickle.dump(newDict, f)


def getFramewisePoseSingleview(dummy, camParamList, mainImgIDList, MetaDictperCam, camMat, beta, dataSet, saveCandImgDir):
    # iterate in [startIdx:endIdx]
    
    for idx, mainImgID in enumerate(mainImgIDList):
        projPtsGT = MetaDictperCam[0][idx]['kpts'][:, 0:2]
        if np.isnan(projPtsGT[0, 0]):
            continue
             
        ImgID = FLAGS.seq + '/' + mainImgID
        
        _, ds = dataSet.createTFExample(itemType='hand', fileIn=ImgID)
        imgRaw = ds.imgRaw
                
        
        # metaperFrame : list of meta_info dict with order [mas, cam1, cam2, cam3]
        
        
        kps3D, poseCoeff, beta, trans, err, _ = lift2Dto3D(projPtsGT, camMat,
                                                        ImgID.split('/')[-1], imgRaw,
                                                        weights=np.ones((21, 1), dtype=np.float32),
                                                        beta=beta,
                                                        rel3DCoordNormGT=None,#fcOut,
                                                        img2DGT=None,
                                                        outDir=saveCandImgDir
                                                        )

        newDict = {'imgID': ImgID,
                   'KPS3D': kps3D,
                   'poseCoeff': poseCoeff,
                   'beta': beta,
                   'trans': trans, 'err': err}

        with open(join(saveCandImgDir, mainImgID.split('/')[-1] +'.pickle'), 'wb') as f:
            pickle.dump(newDict, f)
        
        return newDict




def getFramewisePoseMultiview(dummy, camParamList, mainImgIDList, MetaDictperCam, camMat, beta, dataSet, saveCandImgDir):
    # iterate in [startIdx:endIdx]
    
    for idx, mainImgID in enumerate(mainImgIDList):
        metaperFrame = []
        for camIdx in range(len(camIDset)):
            metaperFrame.append(MetaDictperCam[camIdx][idx])

        ImgID = FLAGS.seq + '/' + mainImgID
        
        _, ds = dataSet.createTFExample(itemType='hand', fileIn=ImgID)
        imgRaw = ds.imgRaw
                
        
        # metaperFrame : list of meta_info dict with order [mas, cam1, cam2, cam3]
        kps3D, poseCoeff, beta, trans, err, _ = lift2Dto3DMultiview(metaperFrame, camParamList, camMat,
                                                        ImgID, imgRaw,
                                                        weights=np.ones((21, 1), dtype=np.float32),
                                                        beta=beta,
                                                        rel3DCoordNormGT=None,#fcOut,
                                                        img2DGT=None,
                                                        outDir=saveCandImgDir
                                                        )

        newDict = {'imgID': ImgID,
                   'KPS3D': kps3D,
                   'poseCoeff': poseCoeff,
                   'beta': beta,
                   'trans': trans, 'err': err}

        with open(join(saveCandImgDir, mainImgID.split('/')[-1] +'.pickle'), 'wb') as f:
            pickle.dump(newDict, f)
        


def main(argv):
    
    resultDir = os.path.join(baseDir, FLAGS.db + '_hand')
    with open(os.path.join(resultDir, "cameraParamsBA.json")) as json_file:
        cameraParams = json.load(json_file)
        cameraParams = {camera: np.array(cameraParams[camera]) for camera in cameraParams}


    intrinsicMatrices = LoadCameraMatrix(os.path.join(resultDir, "220809_cameraInfo.txt"))
    distCoeffs = {}
    distCoeffs["mas"] = LoadDistortionParam(os.path.join(resultDir, "mas_intrinsic.json"))
    distCoeffs["sub1"] = LoadDistortionParam(os.path.join(resultDir, "sub1_intrinsic.json"))
    distCoeffs["sub2"] = LoadDistortionParam(os.path.join(resultDir, "sub2_intrinsic.json"))
    distCoeffs["sub3"] = LoadDistortionParam(os.path.join(resultDir, "sub3_intrinsic.json"))

    camParamList = intrinsicMatrices, cameraParams, distCoeffs
    
    
    saveInitDir = join(baseDir, FLAGS.db, FLAGS.seq, 'handInit')
    if not os.path.exists(saveInitDir):
        os.mkdir(saveInitDir)
    # saveInitDir = join(saveInitDir, FLAGS.camID)
    # if not os.path.exists(saveInitDir):
    #     os.mkdir(saveInitDir)

    saveCandImgDir = join(saveInitDir, 'singleFrameMultiViewFit')
    if not os.path.exists(saveCandImgDir):
        os.mkdir(saveCandImgDir)

    mainCamID = 'mas'
    
    fileListIn = os.listdir(join(OXR_MULTI_CAMERA_DIR, FLAGS.db, FLAGS.seq, 'rgb', mainCamID))
    fileListIn = [join(FLAGS.db, FLAGS.seq, mainCamID, f[:-4]) for f in fileListIn if 'png' in f]
    fileListIn = sorted(fileListIn)
    dataSet = datasetOXRMultiCamera(FLAGS.db, FLAGS.seq, mainCamID, fileListIn=fileListIn)

    # load meta data
    pklDataperCam = []
    mainImgIDList = None
    for camID in camIDset:
        pklFilesList = os.listdir(os.path.join(OXR_MULTI_CAMERA_DIR, FLAGS.db, FLAGS.seq, 'meta', camID))
        pklFilesList = [camID +'/'+ff[:-4] for ff in pklFilesList if 'pkl' in ff]
        pklFilesList = sorted(pklFilesList)
        
        pklDataList = getMeta(pklFilesList)
        pklDataperCam.append(pklDataList)
        
        if camID == mainCamID:
            mainImgIDList = np.copy(pklFilesList)
        
    # get camera matrix
    camMat = dataSet.getCamMat(mainCamID)

    # run independent pose estimation on the candidate frames first. This provides good init for multi frame optimization later
    numThreads = 4 #10
    numCandidateFrames = 8 # len(pklDataperCam[0])
    numFramesPerThread = np.ceil(numCandidateFrames/numThreads).astype(np.uint32)
    procs = []
    
    for proc_index in range(numThreads):
        startIdx = proc_index*numFramesPerThread
        endIdx = min(startIdx+numFramesPerThread,numCandidateFrames)
        
        proc_pklDataperCam = []
        for pklDataList in pklDataperCam:
            proc_pklDataperCam.append(pklDataList[startIdx:endIdx])
                
        args = ([], camParamList, mainImgIDList[startIdx:endIdx], proc_pklDataperCam, camMat, beta, dataSet, saveCandImgDir)
        # proc = mlp.Process(target=getFramewisePoseSingleview, args=args)
        proc = mlp.Process(target=getFramewisePoseMultiview, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()


    ## visualize
    # for i in range(len(fullposeList)):
    #     _, ds = dataSet.createTFExample(itemType='hand', fileIn=perFrameFittingDataList[i]['imgID'])

    #     img = ds.imgRaw

    #     # save the segmentation
    #     # save_annotation.save_annotation(
    #     #     ds.segRaw, saveCandSegDir,
    #     #     perFrameFittingDataList[i]['imgID'].split('/')[-1], add_colormap=True,
    #     #     colormap_type='pascal')

    #     # save the images with kps
    #     manoVis.dump3DModel2DKpsHand(img, mList[i], perFrameFittingDataList[i]['imgID'].split('/')[-1],
    #                                  camMat, est2DJoints=perFrameFittingDataList[i]['KPS2D'], gt2DJoints=None, outDir=saveCandAfterFitDir)

if __name__ == '__main__':
    app.run(main)