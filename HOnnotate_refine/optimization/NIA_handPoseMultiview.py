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

flags.DEFINE_string('db', '230612', 'target db Name') # name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
# flags.DEFINE_string('camID', '0', 'Cam ID')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']

dataset_mix = infUti.datasetMix.OXR_MULTICAMERA

configDir = 'CPMHand'
baseDir = OXR_MULTI_CAMERA_DIR

beta = np.zeros((10,), dtype=np.float32)


def getMeta(files, seq):
    finalPklDataList = []
    ind = 0
    
    while(ind < len(files)):
        file = files[ind]
        resultsDir = join(baseDir, FLAGS.db)

        with open(join(resultsDir, seq, 'meta', file +'.pkl'), 'rb') as f:
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


def getFramewisePoseSingleview(dummy, seq, camParamList, mainImgIDList, MetaDictperCam, camMat, beta, dataSet, saveCandImgDir):
    # iterate in [startIdx:endIdx]
    
    for idx, mainImgID in enumerate(mainImgIDList):
        projPtsGT = MetaDictperCam[0][idx]['kpts'][:, 0:2]
        if np.isnan(projPtsGT[0, 0]):
            continue
             
        ImgID = seq + '/' + mainImgID
        
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




def getFramewisePoseMultiview(dummy, seq, camParamList, mainImgIDList, MetaDictperCam, camMat, beta, dataSet, saveCandImgDir, mainCamID):
    # iterate in [startIdx:endIdx]
    
    for idx, mainImgID in enumerate(mainImgIDList):
        metaperFrame = []
        for camIdx in range(len(camIDset)):
            metaperFrame.append(MetaDictperCam[camIdx][idx])

        ImgID = seq + '/' + mainImgID
        
        _, ds = dataSet.createTFExample(itemType='hand', fileIn=ImgID)
        imgRaw = ds.imgRaw
        otherImgSet = ds.otherImgSet
        
        # metaperFrame : list of meta_info dict with order [mas, cam1, cam2, cam3]
        kps3D, poseCoeff, beta, trans, err, scale = lift2Dto3DMultiview(metaperFrame, camParamList, camMat,
                                                        ImgID, imgRaw,
                                                        weights=np.ones((21, 1), dtype=np.float32),
                                                        beta=beta,
                                                        rel3DCoordNormGT=None,#fcOut,
                                                        img2DGT=None,
                                                        outDir=saveCandImgDir,
                                                        otherImgSet=otherImgSet, mainCamID=mainCamID
                                                        )

        newDict = {'imgID': ImgID,
                   'KPS3D': kps3D,
                   'poseCoeff': poseCoeff,
                   'beta': beta,
                   'trans': trans, 
                   'err': err, 
                   'scale': scale,
                   'meta': metaperFrame}

        with open(join(saveCandImgDir, mainImgID.split('/')[-1] +'.pickle'), 'wb') as f:
            pickle.dump(newDict, f)
        


def main(argv):
    
    
    # camIDset = ['mas', 'sub1', 'sub2', 'sub3']
    mainCamID = 1
    
    resultDir = os.path.join(baseDir, FLAGS.db + '_cam')
    with open(os.path.join(resultDir, "cameraParamsBA.json")) as json_file:
        cameraParams = json.load(json_file)
        cameraParams = {camera: np.array(cameraParams[camera]) for camera in cameraParams}

    
    intrinsicMatrices = LoadCameraMatrix(os.path.join(resultDir, "230612_cameraInfo.txt"))
    distCoeffs = {}
    distCoeffs["mas"] = LoadDistortionParam(os.path.join(resultDir, "mas_intrinsic.json"))
    distCoeffs["sub1"] = LoadDistortionParam(os.path.join(resultDir, "sub1_intrinsic.json"))
    distCoeffs["sub2"] = LoadDistortionParam(os.path.join(resultDir, "sub2_intrinsic.json"))
    distCoeffs["sub3"] = LoadDistortionParam(os.path.join(resultDir, "sub3_intrinsic.json"))

    
    ### no translation in mano space ###
    for camID in camIDset:
        cameraParams[camID] = cameraParams[camID].reshape(3, 4)
        cameraParams[camID] = cameraParams[camID]

    camParamList = intrinsicMatrices, cameraParams, distCoeffs
    
    ### start handpose optimization for each sequences
    rootDir = os.path.join(baseDir, FLAGS.db)
    total_seq = len(os.listdir(rootDir))
    
    for seqIdx, seq in enumerate(sorted(os.listdir(rootDir))):
        d = os.path.join(rootDir, seq)
        if os.path.isdir(d):            
            print("(%s in %s) : %s" % (seqIdx, total_seq, seq))
    
            saveInitDir = join(baseDir, FLAGS.db, seq, 'handInit')
            if not os.path.exists(saveInitDir):
                os.mkdir(saveInitDir)

            saveCandImgDir = join(saveInitDir, 'singleFrameMultiViewFit')
            if not os.path.exists(saveCandImgDir):
                os.mkdir(saveCandImgDir)

            
            fileListIn = os.listdir(join(OXR_MULTI_CAMERA_DIR, FLAGS.db, seq, 'rgb_crop', camIDset[mainCamID]))
            fileListIn = [join(FLAGS.db, seq, camIDset[mainCamID], f[:-4]) for f in fileListIn if 'png' in f]
            fileListIn = sorted(fileListIn)
            dataSet = datasetOXRMultiCamera(FLAGS.db, seq, camIDset[mainCamID], fileListIn=fileListIn)

            # load meta data
            pklDataperCam = []
            mainImgIDList = None
            for camID in camIDset:
                pklFilesList = os.listdir(os.path.join(OXR_MULTI_CAMERA_DIR, FLAGS.db, seq, 'meta', camID))
                pklFilesList = [camID +'/'+ff[:-4] for ff in pklFilesList if 'pkl' in ff]
                pklFilesList = sorted(pklFilesList)
                
                pklDataList = getMeta(pklFilesList, seq)
                pklDataperCam.append(pklDataList)
                
                if camID == camIDset[mainCamID]:
                    mainImgIDList = np.copy(pklFilesList)
                
            # get camera matrix
            camMat = dataSet.getCamMat(camIDset[mainCamID])

            # run independent pose estimation on the candidate frames first. This provides good init for multi frame optimization later
            numThreads = 5
            numCandidateFrames = len(pklDataperCam[0])
            numFramesPerThread = np.ceil(numCandidateFrames/numThreads).astype(np.uint32)
            procs = []
            
            for proc_index in range(numThreads):
                startIdx = proc_index*numFramesPerThread
                endIdx = min(startIdx+numFramesPerThread,numCandidateFrames)
                
                proc_pklDataperCam = []
                for pklDataList in pklDataperCam:
                    proc_pklDataperCam.append(pklDataList[startIdx:endIdx])
                        
                args = ([], seq, camParamList, mainImgIDList[startIdx:endIdx], proc_pklDataperCam, camMat, beta, dataSet, saveCandImgDir, mainCamID)
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


    """
    [visualize code sample for kps3D in .pickle]
    ### required info
    # mainCampose(extrinsic) : main camera during pose optimization(fixed, currently sub1)
    # camPose(extrinsic) : (3,4) extrinsic parameter of target camera for visualize
    # camMat(intrinsic) : (3,3) intrinsic parameter of target camera for visualize
    # CamID : target camera ID, index of camIDset = ['mas', 'sub1', 'sub2', 'sub3']
    
    
    kps3D = sample['KPS3D']
    scale = sample['scale']    
    img2bb = sample['meta'][CamID]['img2bb']
    
    targetPose = kps3D * scale
        
    # original projection   
    mano4Dcamera = np.concatenate([targetPose, np.ones((21, 1))], axis=1)
    projection = np.concatenate((mainCamPose, h), 0)
    mano4Dworld = np.linalg.inv(projection).dot(mano4Dcamera.T).T  
    
     ### project mano model pts to each cam     
    mano4Dcamera2 = mano4Dworld.dot(camPose.reshape(3,4).T)
    
    ## already transformed jointmap
    projPts = utilsEval.cv2ProjectPoints(mano4Dcamera2, camMat, False)  #[jointsMap]
    
    # add crop on bb
    uv1 = np.concatenate((projPts, np.ones_like(projPts[:, :1])), 1)
    projPts = np.dot(img2bb, np.transpose(uv1)).T
        
    axEst = fig.add_subplot(2, 2, 3)
    imgOutEst = utilsEval.showHandJoints(img.copy(), np.copy(projPts).astype(np.float32), estIn=None,
                                         filename=None,
                                         upscale=1, lineThickness=3)
    axEst.imshow(imgOutEst[:, :, [2, 1, 0]])
    
    """
    
    
    
if __name__ == '__main__':
    app.run(main)