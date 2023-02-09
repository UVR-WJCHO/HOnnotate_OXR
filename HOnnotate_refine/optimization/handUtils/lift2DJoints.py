import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from imageio import imread, imsave

import argparse, json, yaml
import cv2
from mano.webuser.verts import verts_core
import chumpy as ch
from sklearn.preprocessing import normalize
import handUtils.manoHandVis as manoVis
from eval import utilsEval
import open3d

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)

from chumpy.ch_ops import UnaryElemwise as UE




delta = 50
class clipIden(UE):
    _r = lambda self, x: (np.abs(x) < delta) * x + 0.0#(np.abs(x) >= delta)*(0.01*(x))#-delta) + delta)
    _d = lambda self, x: (np.abs(x) < delta) * 1 + 0.0#(np.abs(x) >= delta) * 0.01

render = True
cntr = 0

jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

h = np.array([[0,0,0,1]])

camIDset = ['mas', 'sub1', 'sub2', 'sub3']


def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class Constraints():
    def __init__(self):
        self.thetaLimits()

    def thetaLimits(self):
        MINBOUND = -5.
        MAXBOUND = 5.
        self.validThetaIDs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                                       30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 44, 46, 47], dtype=np.int32)
        # self.invalidThetaIDs = np.array([7, 9, 10, 12, 16, 18, 19, 24,
        #                                25, 27, 28, 34, 36, 37, 39, 43, 45], dtype=np.int32)
        invalidThetaIDsList = []
        for i in range(48):
            if i not in self.validThetaIDs:
                invalidThetaIDsList.append(i)
        self.invalidThetaIDs = np.array(invalidThetaIDsList)

        self.minThetaVals = np.array([MINBOUND, MINBOUND, MINBOUND, # global rot
                            0, -0.15, 0.1, -0.3, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # index
                            MINBOUND, -0.15, 0.1, -0.5, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # middle
                            -1.5, -0.15, -0.1, MINBOUND, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # pinky
                            -0.5, -0.25, 0.1, -0.4, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # ring
                            MINBOUND, -0.83, -0.0, -0.15, MINBOUND, 0, MINBOUND, -0.5, -1.57, ]) # thumb

        self.maxThetaVals = np.array([MAXBOUND, MAXBOUND, MAXBOUND, #global
                            0.45, 0.2, 1.8, 0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # index
                            MAXBOUND, 0.15, 2.0, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # middle
                            -0.2, 0.15, 1.6, MAXBOUND, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # pinky
                            -0.4, 0.10, 1.6, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # ring
                            MAXBOUND, 0.66, 0.5, 1.6, MAXBOUND, 0.5, MAXBOUND, 0, 1.08]) # thumb

        # self.minThetaVals = np.array([MINBOUND, MINBOUND, MINBOUND,  # global rot
        #                               0, -0.15, 0.1, -0.3, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # index
        #                               MINBOUND, -0.15, 0.1, -0.5, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # middle
        #                               -1.0, -0.15, -0.1, MINBOUND, -0.5, -0.0, MINBOUND, MINBOUND, 0,  # pinky
        #                               -0.5, -0.25, 0.1, -0.4, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # ring
        #                               0.0, -0.83, -0.0, -0.15, MINBOUND, 0, MINBOUND, -0.5, -1.57, ])  # thumb
        #
        # self.maxThetaVals = np.array([MAXBOUND, MAXBOUND, MAXBOUND,  # global
        #                               0.45, 0.2, 1.8, 0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # index
        #                               MAXBOUND, 0.15, 2.0, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # middle
        #                               -0.2, 0.6, 1.6, MAXBOUND, 0.6, 2.0, MAXBOUND, MAXBOUND, 1.25,  # pinky
        #                               -0.4, 0.10, 1.8, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # ring
        #                               2.0, 0.66, 0.5, 1.6, MAXBOUND, 0.5, MAXBOUND, 0, 1.08])  # thumb

        self.fullThetaMat = np.zeros((48, len(self.validThetaIDs)), dtype=np.float32)  #48x25
        for i in range(len(self.validThetaIDs)):
            self.fullThetaMat[self.validThetaIDs[i], i] = 1.0

    def getHandJointConstraints(self, theta, isValidTheta=False):
        '''
        get constraints on the joint angles when input is theta vector itself (first 3 elems are NOT global rot)
        :param theta: Nx45 tensor if isValidTheta is False and Nx25 if isValidTheta is True
        :param isValidTheta:
        :return:
        '''

        if not isValidTheta:
            assert (theta.shape)[-1] == 45
            validTheta = theta[self.validThetaIDs[3:] - 3]
        else:
            assert (theta.shape)[-1] == len(self.validThetaIDs[3:])
            validTheta = theta

        phyConstMax = (ch.maximum(self.minThetaVals[self.validThetaIDs[3:]] - validTheta, 0))
        phyConstMin = (ch.maximum(validTheta - self.maxThetaVals[self.validThetaIDs[3:]], 0))

        return phyConstMin, phyConstMax


def getHandModel():
    globalJoints = ch.zeros((45,))
    globalBeta = ch.zeros((10,))
    chRot = ch.zeros((3,))
    chTrans = ch.array([0., 0., 0.5])

    fullpose = ch.concatenate([chRot, globalJoints], axis=0)
    m = load_model_withInputs(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../mano/models/MANO_RIGHT.pkl'), fullpose, chTrans,
                              globalBeta,
                              ncomps=15, flat_hand_mean=True)

    return m, chRot, globalJoints, chTrans, globalBeta

def getHandModelPoseCoeffs(numComp):
    m = load_model_1(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../mano/models/MANO_RIGHT.pkl'), ncomps=numComp)
    m.trans[:] = np.array([0., 0., 0.5])
    

    return m, m.pose, m.betas, m.trans, m.fullpose


def lift2Dto3D(projPtsGT, camMat, filename, img, JVis=np.ones((21,), dtype=np.float32), trans=None, beta=None, wrist3D = None, withPoseCoeff=True,
               weights=1.0, relDepGT=None, rel3DCoordGT = None, rel3DCoordNormGT = None, img2DGT = None, outDir = None,
               poseCoeffInit=None,
               transInit=None, betaInit=None):

    loss = {}

    if withPoseCoeff:
        numComp = 30
        m, poseCoeffCh, betaCh, transCh, fullposeCh = getHandModelPoseCoeffs(numComp)

        if poseCoeffInit is not None:
            poseCoeffCh[:] = poseCoeffInit

        if transInit is not None:
            transCh[:] = transInit

        if betaInit is not None:
            betaCh[:] = betaInit

        freeVars = [poseCoeffCh]
        if beta is None:
            freeVars = freeVars + [betaCh]
            loss['shape'] = 1e2 * betaCh
        else:
            betaCh[:] = beta

        if trans is None:
            freeVars = freeVars + [transCh]
        else:
            transCh[:] = trans

        # loss['pose'] = 0.5e2 * poseCoeffCh[3:]/stdPCACoeff[:numComp]

        thetaConstMin, thetaConstMax = Constraints().getHandJointConstraints(fullposeCh[3:])
        loss['constMin'] = 5e2 * thetaConstMin
        loss['constMax'] = 5e2 * thetaConstMax
        loss['invalidTheta'] = 1e3 * fullposeCh[Constraints().invalidThetaIDs]

    else:
        m, rotCh, jointsCh, transCh, betaCh = getHandModel()

        thetaConstMin, thetaConstMax = Constraints().getHandJointConstraints(jointsCh)
        loss['constMin'] = 5e3 * thetaConstMin
        loss['constMax'] = 5e3 * thetaConstMax
        validTheta = jointsCh[Constraints().validThetaIDs[3:] - 3]

        freeVars = [validTheta, rotCh]

        if beta is None:
            freeVars = freeVars + [betaCh]
            loss['shape'] = 0.5e2 * betaCh
        else:
            betaCh[:] = beta

        if trans is None:
            freeVars = freeVars + [transCh]
        else:
            transCh[:] = trans

    if relDepGT is not None:
        relDepPred = m.J_transformed[:,2] - m.J_transformed[0,2]
        loss['relDep'] = (relDepPred - relDepGT) * weights[:,0] * 5e1

    if rel3DCoordGT is not None:
        rel3DCoordPred = m.J_transformed - m.J_transformed[0:1,:]
        loss['rel3DCoord'] = (rel3DCoordPred - rel3DCoordGT) * np.tile(weights[:,0:1], [1,3]) * 5e1

    if rel3DCoordNormGT is not None:
        rel3DCoordPred = m.J_transformed[jointsMap][1:,:] - m.J_transformed[jointsMap][0:1, :]

        rel3DCoordPred = rel3DCoordPred/ch.expand_dims(ch.sqrt(ch.sum(ch.square(rel3DCoordPred), axis=1)), axis=1)
        loss['rel3DCoordNorm'] = (1. - ch.sum(rel3DCoordPred*rel3DCoordNormGT, axis=1))*1e4

        # loss['rel3DCoordNorm'] = \
        #     (rel3DCoordNormGT*ch.expand_dims(ch.sum(rel3DCoordPred*rel3DCoordNormGT, axis=1), axis=1) - rel3DCoordPred) * 1e2#5e2



    # pts3D = np.copy(m.J_transformed)
    projPts = utilsEval.chProjectPoints(m.J_transformed, camMat, False)[jointsMap]
    projPts_debug = np.copy(projPts)
    
    JVis = np.tile(np.expand_dims(JVis, 1), [1, 2])
    loss['joints2D'] = (projPts - projPtsGT) * JVis * weights * 1e0
    loss['joints2DClip'] = clipIden(projPts - projPtsGT) * JVis * weights * 1e1


    if wrist3D is not None:
        dep = wrist3D[2]
        if dep<0:
            dep = -dep
        loss['wristDep'] = (m.J_transformed[0,2] - dep)*1e2


    # vis_mesh(m)



    render = False

    def cbPass(_):

        pass
        # print(loss['joints'].r)



    print(filename)
    warnings.simplefilter('ignore')


    loss['joints2D'] = loss['joints2D'] * 1e1/ weights # dont want to use confidence now

    if True:
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cbPass if render else cbPass, method='dogleg', options={'maxiter': 20})
    else:
        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT,
                                     outDir=outDir)

        freeVars = [poseCoeffCh[:3], transCh]
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cbPass, method='dogleg', options={'maxiter': 20})

        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT,
                                     outDir=outDir)
        freeVars = [poseCoeffCh[3:]]
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cb if render else cbPass, method='dogleg', options={'maxiter': 20})

        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT,
                                     outDir=outDir)
        freeVars = [poseCoeffCh, transCh]
        if beta is None:
            freeVars = freeVars + [betaCh]
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cb if render else cbPass, method='dogleg', options={'maxiter': 20})


    if False:
        open3dVisualize(m)
    else:
        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT, outDir=outDir)






    # vis_mesh(m)

    joints3D = m.J_transformed.r[jointsMap]

    # print(betaCh.r)
    # print((relDepPred.r - relDepGT))

    return joints3D, poseCoeffCh.r.copy(), betaCh.r.copy(), transCh.r.copy(), loss['joints2D'].r.copy(), m.r.copy()


def lift2Dto3DMultiview(metaperFrame, camParamList, camMat, filename, img, JVis=np.ones((21,), dtype=np.float32), trans=None, beta=None, wrist3D = None, withPoseCoeff=True,
               weights=1.0, relDepGT=None, rel3DCoordGT = None, rel3DCoordNormGT = None, img2DGT = None, outDir = None,
               poseCoeffInit=None, transInit=None, betaInit=None, otherImgSet=None, mainCamID=None):


    
    # metaperFrame = [meta_mas, meta_sub1, meta_sub2, meta_sub3]
    # meta = {'bb': bb, 'img2bb': np.float32(img2bb), 'bb2img': np.float32(bb2img), 'kpts': np.float32(processed_kpts)}  
    
    intrinsicMatrices, extrinsicMatrices, distCoeffs = camParamList
    
    loss = {}

    if withPoseCoeff:
        numComp = 30
        m, poseCoeffCh, betaCh, transCh, fullposeCh = getHandModelPoseCoeffs(numComp)

        if poseCoeffInit is not None:
            poseCoeffCh[:] = poseCoeffInit

        if transInit is not None:
            transCh[:] = transInit

        if betaInit is not None:
            betaCh[:] = betaInit

        freeVars = [poseCoeffCh]
        if beta is None:
            freeVars = freeVars + [betaCh]
            loss['shape'] = 1e2 * betaCh
        else:
            betaCh[:] = beta

        if trans is None:
            freeVars = freeVars + [transCh]
        else:
            transCh[:] = trans
            

        # loss['pose'] = 0.5e2 * poseCoeffCh[3:]/stdPCACoeff[:numComp]

        thetaConstMin, thetaConstMax = Constraints().getHandJointConstraints(fullposeCh[3:])
        loss['constMin'] = 5e2 * thetaConstMin
        loss['constMax'] = 5e2 * thetaConstMax
        loss['invalidTheta'] = 1e3 * fullposeCh[Constraints().invalidThetaIDs]

    else:
        m, rotCh, jointsCh, transCh, betaCh = getHandModel()

        thetaConstMin, thetaConstMax = Constraints().getHandJointConstraints(jointsCh)
        loss['constMin'] = 5e3 * thetaConstMin
        loss['constMax'] = 5e3 * thetaConstMax
        validTheta = jointsCh[Constraints().validThetaIDs[3:] - 3]

        freeVars = [validTheta, rotCh]

        if beta is None:
            freeVars = freeVars + [betaCh]
            loss['shape'] = 0.5e2 * betaCh
        else:
            betaCh[:] = beta

        if trans is None:
            freeVars = freeVars + [transCh]
        else:
            transCh[:] = trans

    if relDepGT is not None:
        relDepPred = m.J_transformed[:,2] - m.J_transformed[0,2]
        loss['relDep'] = (relDepPred - relDepGT) * weights[:,0] * 5e1

    if rel3DCoordGT is not None:
        rel3DCoordPred = m.J_transformed - m.J_transformed[0:1,:]
        loss['rel3DCoord'] = (rel3DCoordPred - rel3DCoordGT) * np.tile(weights[:,0:1], [1,3]) * 5e1

    if rel3DCoordNormGT is not None:
        rel3DCoordPred = m.J_transformed[jointsMap][1:,:] - m.J_transformed[jointsMap][0:1, :]

        rel3DCoordPred = rel3DCoordPred/ch.expand_dims(ch.sqrt(ch.sum(ch.square(rel3DCoordPred), axis=1)), axis=1)
        loss['rel3DCoordNorm'] = (1. - ch.sum(rel3DCoordPred*rel3DCoordNormGT, axis=1))*1e4

    scale = ch.array(np.array([1.]))
    freeVars = freeVars + [scale]
    
    # initPose = ch.array(np.ones((21, 3)))
    # freeVars = [initPose]
   
    ### check if GT exists, for debugging, return 0 if any of GT is missing
    for camIdx in range(len(camIDset)):
        projPtsGT = metaperFrame[camIdx]['kpts'][:, 0:2]
        if np.isnan(projPtsGT[0, 0]):
            joints3D = np.zeros((21, 3))
            return joints3D, poseCoeffCh.r.copy(), betaCh.r.copy(), transCh.r.copy(), 0, None
    
   
    ### mano output is on world coordinate? wrong results...why?
    # mano4Dworld = ch.concatenate([m.J_transformed, np.ones((21, 1))], axis=1)
    
    input_xyz = m.J_transformed * scale
    # input_xyz = initPose
    
    ### mano output is on camera coordinate? correct results only for single cam
    mano4Dcamera = ch.concatenate([input_xyz, np.ones((21, 1))], axis=1)
    projection = ch.concatenate((extrinsicMatrices[camIDset[mainCamID]].reshape(3,4), h), 0)
    mano4Dworld = ch.linalg.inv(projection).dot(mano4Dcamera.T).T   
    

    # (3d homogeneous point in image coordinate) = (intrinsic) * (extrinsic) * (3d homogeneous point in world coordinate)
    # (3d homogeneous point in camera coordinate) = (extrinsic) * (3d homogeneous point in world coordinate)
 
    ######### debug log #########
    """
    projPtsGT_mas = metaperFrame[0]['kpts'][:, 0:2]
    projPtsGT_sub1 = metaperFrame[1]['kpts'][:, 0:2]
    
    ex_mas = extrinsicMatrices['mas'].reshape(3,4)      
    ex_sub1 = extrinsicMatrices['sub1'].reshape(3,4)
    
    ## [ex_mas]
    ## array([ 0.99651512, -0.00185681,  0.00172505,  1.06138909])
    ## array([-3.77008816e-04,  9.94952911e-01,  1.74648926e-04,  2.17816869e-01])
    ## array([-0.00515595, -0.00294564,  1.00277799,  1.61680288])
        
    ## [ex_sub1]    
    ## array([ 5.56268556e-01, -1.83593177e-01,  8.07564738e-01, -3.57132190e+02])
    ## array([  -0.27903086,    0.8662851 ,    0.39915127, -179.61030311])
    ## array([ -0.78594016,  -0.44296072,   0.43649803, 320.80500681])

    mano4Dcamera2_mas = mano4Dworld.dot(extrinsicMatrices['mas'].reshape(3,4).T)[:, :3]
    mano4Dcamera2_sub1 = mano4Dworld.dot(extrinsicMatrices['sub1'].reshape(3,4).T)[:, :3]
        
    ## mano4Dcamera2_mas : array([[0.09566993, 0.00638343, 0.50618631], ...])
    ## mano4Dcamera2_sub1 : array([-358.53030295, -179.96887255,  321.17378001], ... 1 이하 range 유사한 value 반복)
    
    ## 다른 cam인것 감안해도 sub1에서의 coord가 거의 점 사이즈의 손으로 보임.
        
    projPts_mas = utilsEval.chProjectPoints(mano4Dcamera2_mas, intrinsicMatrices['mas'], False)[jointsMap]
    projPts_sub1 = utilsEval.chProjectPoints(mano4Dcamera2_sub1, intrinsicMatrices['sub1'], False)[jointsMap]
    
    
    #2Dproj value도 sub1은 1 이하 value 차이로 변화 joint u value : (-43.71582987, -43.61781355, -43.55728923, ...))
    """
    #############################
        
    loss_joint2D = 0
    loss_joint2DClip = 0
    JVis = np.tile(np.expand_dims(JVis, 1), [1, 2])
            
    for camIdx, cam in enumerate(camIDset):  
        # if camIdx > 1:
        #     continue
        ### estimated 2D keyPts on each cam
        projPtsGT = metaperFrame[camIdx]['kpts_crop'][:, 0:2]
        
        if np.isnan(projPtsGT[0, 0]):
            continue
        
        # ### project mano model pts to each cam     
        mano4Dcamera2 = mano4Dworld.dot(extrinsicMatrices[cam].reshape(3,4).T)#[:, :3]
        # ## need jointMap if it from mano model
        projPts = utilsEval.chProjectPoints(mano4Dcamera2, intrinsicMatrices[cam], False)[jointsMap]
        
        
        # xyz_cam = ch.matmul(mano4Dworld, extrinsicMatrices[cam].T)
        # xyz_cam = xyz_cam[:, :3] / xyz_cam[:, -1:]  # xyz_cam : (-0.38 0.16, 1)

        # uv = ch.matmul(xyz_cam, intrinsicMatrices[cam].T)
        # projPts = uv[:, :2] / uv[:, -1:]
        
        
    
        ### transform pts with each cropped bb
        img2bb = metaperFrame[camIdx]['img2bb']
        uv1 = ch.concatenate((projPts, ch.ones_like(projPts[:, :1])), 1)
        projPts = ch.transpose(ch.dot(img2bb, ch.transpose(uv1)))
                
        loss_key = 'joints2D_' + str(camIdx)
        loss[loss_key] = (projPts - projPtsGT) * JVis * weights * 1e0
       

    if wrist3D is not None:
        dep = wrist3D[2]
        if dep<0:
            dep = -dep
        loss['wristDep'] = (m.J_transformed[0,2] - dep)*1e2


    # vis_mesh(m)


    render = False

    def cbPass(_):

        pass
        # print(loss['joints'].r)


    print(filename)
    filename = filename.split('/')[-1]
    
    warnings.simplefilter('ignore')


    # loss['joints2D_0'] = loss['joints2D_0'] * 1e1/ weights # dont want to use confidence now
    # loss['joints2D_1'] = loss['joints2D_1'] * 1e1/ weights # dont want to use confidence now
    # loss['joints2D_2'] = loss['joints2D_2'] * 1e1/ weights # dont want to use confidence now
    # loss['joints2D_3'] = loss['joints2D_3'] * 1e1/ weights # dont want to use confidence now
    
    if True:
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cbPass if render else cbPass, method='dogleg', options={'maxiter': 50})
    else:
        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT,
                                     outDir=outDir)

        freeVars = [poseCoeffCh[:3], transCh]
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cbPass, method='dogleg', options={'maxiter': 20})

        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT,
                                     outDir=outDir)
        freeVars = [poseCoeffCh[3:]]
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cb if render else cbPass, method='dogleg', options={'maxiter': 20})

        manoVis.dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints=projPtsGT, gt2DJoints=img2DGT,
                                     outDir=outDir)
        freeVars = [poseCoeffCh, transCh]
        if beta is None:
            freeVars = freeVars + [betaCh]
        ch.minimize({k: loss[k] for k in loss.keys() if k != 'joints2DClip'}, x0=freeVars,
                    callback=cb if render else cbPass, method='dogleg', options={'maxiter': 20})

    
    if False:
        open3dVisualize(m)
    else:
        # mainprojPtsGT = metaperFrame[mainCamID]['kpts'][:, 0:2]
        # mainimg2bb = metaperFrame[mainCamID]['img2bb']
        # manoVis.dump3DModel2DKpsHand_forCrop(img, m, filename, camMat, img2bb=mainimg2bb, gt2DJoints=mainprojPtsGT, outDir=outDir)
        
        # debugPose = initPose 
        debugPose = None
        
        for idx, cam in enumerate(camIDset):
            if idx == mainCamID:
                mainprojPtsGT = metaperFrame[mainCamID]['kpts_crop'][:, 0:2]
                mainimg2bb = metaperFrame[mainCamID]['img2bb']
                maincamPose = extrinsicMatrices[camIDset[mainCamID]].reshape(3,4)
                
                manoVis.dump3DModel2DKpsHand_forCrop(img, m, filename, camMat, 
                                                     img2bb=mainimg2bb, gt2DJoints=mainprojPtsGT, outDir=outDir, 
                                                     camPose=maincamPose, mainCamPose=maincamPose, debug=debugPose, scale=scale)
            else:
                otherimg = otherImgSet[cam]
                othercamMat = intrinsicMatrices[cam]
                
                maincamPose = extrinsicMatrices[camIDset[mainCamID]].reshape(3,4)
                othercamPose = extrinsicMatrices[cam].reshape(3,4)
                
                otherprojPtsGT = metaperFrame[idx]['kpts_crop'][:, 0:2]
                otherimg2bb = metaperFrame[idx]['img2bb']
                
                otherFilename = str(cam) + '_' + filename.split('_')[-1]
                manoVis.dump3DModel2DKpsHand_forCrop(otherimg, m, otherFilename, othercamMat, 
                                                     img2bb=otherimg2bb, gt2DJoints=otherprojPtsGT, 
                                                     outDir=outDir, camPose=othercamPose, mainCamPose=maincamPose, debug=debugPose, scale=scale)
     
                
                



    # vis_mesh(m)

    joints3D = m.J_transformed.r[jointsMap]

    # print(betaCh.r)
    # print((relDepPred.r - relDepGT))

    err = 0 #loss['joints2D_0'].r + loss['joints2D_1'].r + loss['joints2D_2'].r + loss['joints2D_3'].r
    
    return joints3D, poseCoeffCh.r.copy(), betaCh.r.copy(), transCh.r.copy(), err, scale



def ready_arguments(fname_or_dict, posekey4vposed='pose', shared_args=None, chTrans=None, chBetas=None):
    import numpy as np
    import pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult
    from mano.webuser.posemapper import posemap

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    if chTrans is not None:
        dd['trans'] = chTrans

    if chTrans is not None:
        dd['betas'] = chBetas

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J', 'fullpose', 'pose_dof']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            if shared_args is not None and s in shared_args:
                dd[s] = shared_args[s]
            else:
                dd[s] = ch.array(dd[s])

    assert(posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))

    return dd

def load_model_1(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None, shared_args=None, optwrt='pose_coeff', relRot=np.eye(3), relTrans=np.array([0.0, 0.0, 0.0])):
    ''' This model loads the fully articulable HAND SMPL model,
    and replaces the pose DOFS by ncomps from PCA'''

    import numpy as np
    import chumpy as ch
    import pickle
    import scipy.sparse as sp
    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
        with open(fname_or_dict, 'rb') as f:
            smpl_data = pickle.load(f, encoding='latin1')
    else:
        smpl_data = fname_or_dict

    rot = 3  # for global orientation!!!
    dof = 20

    # smpl_data['hands_components'] = np.eye(45)
    from  sklearn.preprocessing import normalize
    smpl_data['hands_components'] = normalize(smpl_data['hands_components'], axis=1)
    hands_components = smpl_data['hands_components']
    std = np.linalg.norm(hands_components, axis=1)
    hands_mean       = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
    hands_coeffs     = smpl_data['hands_coeffs'][:, :ncomps]

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    if shared_args is not None and 'pose_coeffs' in shared_args:
        pose_coeffs = ch.zeros(rot + selected_components.shape[0])
        pose_coeffs[:len(shared_args['pose_coeffs'])] = shared_args['pose_coeffs']
    else:
        pose_coeffs = ch.zeros(rot + selected_components.shape[0])
    full_hand_pose = pose_coeffs[rot:(rot+ncomps)].dot(selected_components)

    smpl_data['fullpose'] = ch.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))
    pose_dof = ch.zeros(rot + dof)

    smpl_data['pose'] = pose_coeffs
    smpl_data['pose_dof'] = pose_dof

    Jreg = smpl_data['J_regressor']
    if not sp.issparse(Jreg):
        smpl_data['J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    # slightly modify ready_arguments to make sure that it uses the fullpose
    # (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(smpl_data, posekey4vposed='fullpose')

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }
    # print(dd['J'].r)

    result_previous, meta = verts_core(**args)
    result_noRel = result_previous + dd['trans'].reshape((1, 3))
    result = result_noRel.dot(relRot) + relTrans
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if hasattr(result, 'Jtr'):
        result.J_transformed = (result.Jtr + dd['trans'].reshape((1, 3))).dot(relRot) + relTrans

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template

    return result

def load_model_withInputs_poseCoeffs(fname_or_dict, chRot, chPoseCoeff, chTrans, chBetas,
                          ncomps=6, flat_hand_mean=False, v_template=None, shared_args=None,):

    import numpy as np
    import chumpy as ch
    import pickle
    import scipy.sparse as sp
    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
        # smpl_data = pickle.load(open(fname_or_dict))
        smpl_data = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        smpl_data = fname_or_dict

    rot = 3  # for global orientation!!!
    dof = 20

    # smpl_data['hands_components'] = np.eye(45)
    from  sklearn.preprocessing import normalize
    smpl_data['hands_components'] = normalize(smpl_data['hands_components'], axis=1)
    hands_components = smpl_data['hands_components']
    hands_mean       = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
    hands_coeffs     = smpl_data['hands_coeffs'][:, :ncomps]

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    pose_coeffs = ch.concatenate([chRot, chPoseCoeff], axis=0)
    full_hand_pose = pose_coeffs[rot:(rot+ncomps)].dot(selected_components)

    smpl_data['fullpose'] = ch.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))

    smpl_data['pose'] = pose_coeffs

    Jreg = smpl_data['J_regressor']
    if not sp.issparse(Jreg):
        smpl_data['J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    # slightly modify ready_arguments to make sure that it uses the fullpose
    # (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(smpl_data, posekey4vposed='fullpose', shared_args=shared_args, chTrans=chTrans, chBetas=chBetas)

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }
    # print(dd['J'].r)

    result_previous, meta = verts_core(**args)
    result_noRel = result_previous + dd['trans'].reshape((1, 3))
    result = result_noRel
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if hasattr(result, 'Jtr'):
        result.J_transformed = (result.Jtr + dd['trans'].reshape((1, 3)))

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template

    return result


def getHandModelPoseCoeffsMultiFrame(numComp, numFrames, isOpenGLCoord):
    chGlobalPoseCoeff = ch.zeros((numComp,))
    chGlobalBeta = ch.zeros((10,))

    chRotList = []
    chTransList = []
    mList = []

    for i in range(numFrames):
        chRot = ch.zeros((3,))
        if isOpenGLCoord:
            chTrans = ch.array([0., 0., -0.5])
        else:
            chTrans = ch.array([0., 0., 0.5])
        chRotList.append(chRot)
        chTransList.append(chTrans)

        m = load_model_withInputs_poseCoeffs(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../mano/models/MANO_RIGHT.pkl'),
                                             chRot=chRot,
                                             chTrans=chTrans,
                                             chPoseCoeff = chGlobalPoseCoeff,
                                             chBetas=chGlobalBeta,
                                             ncomps=numComp)
        mList.append(m)

    return mList, chRotList, chGlobalPoseCoeff, chTransList, chGlobalBeta


def lift2Dto3DMultiFrame(projPtsGT, camMat, filename, JVis=np.ones((21,), dtype=np.float32), trans=None, beta=None,
                         wrist3D = None, withPoseCoeff=True,
                         weights=None, relDepGT=None, rel3DCoordGT = None, isOpenGLCoord=False,
                         transInit = None, rotInit = None, globalPoseCoeffInit = None, betaInit = None):
    '''

    :param projPtsGT:
    :param camMat:
    :param filename:
    :param JVis:
    :param trans:
    :param beta:
    :param wrist3D:
    :param withPoseCoeff:
    :param weights: 21x1 array
    :param relDepGT:
    :param rel3DCoordGT: always in opencv
    :param isOpenGLCoord:
    :return: always in opencv
    '''

    loss = {}


    numFrames = projPtsGT.shape[0]
    if weights is None:
        weights = [np.ones((21,1), dtype=np.float32)]*numFrames

    numComp = 30
    mList, chRotList, chGlobalPoseCoeff, chTransList, chGlobalBeta = getHandModelPoseCoeffsMultiFrame(numComp, numFrames, isOpenGLCoord)

    freeVars = [chGlobalPoseCoeff]
    if beta is None:
        freeVars = freeVars + [chGlobalBeta]
        loss['shape'] = 1e2 * chGlobalBeta
    else:
        chGlobalBeta[:] = beta

    if betaInit is not None:
        chGlobalBeta[:] = betaInit

    if trans is None:
        freeVars = freeVars + chTransList
    else:
        for i, t in enumerate(chTransList):
            t[:] = trans[i]

    if transInit is not None:
        for i in range(len(chTransList)):
            chTransList[i][:] = transInit[i]

    if rotInit is not None:
        for i in range(len(chRotList)):
            chRotList[i][:] = rotInit[i]

    if globalPoseCoeffInit is not None:
        chGlobalPoseCoeff[:] = globalPoseCoeffInit

    freeVars = freeVars + chRotList
    # loss['pose'] = 0.5e2 * poseCoeffCh[3:]/stdPCACoeff[:numComp]

    fullposeCh = mList[0].fullpose ##

    thetaConstMin, thetaConstMax = Constraints().getHandJointConstraints(fullposeCh[3:])
    loss['constMin'] = 5e2 * thetaConstMin
    loss['constMax'] = 5e2 * thetaConstMax
    loss['invalidTheta'] = 5e2 * fullposeCh[Constraints().invalidThetaIDs]

    if relDepGT is not None:
        for i in range(numFrames):
            relDepPred = mList[i].J_transformed[:,2] - mList[i].J_transformed[0,2]
            loss['relDep_%d'%(i)] = (relDepPred - relDepGT[i]) * weights[:,0] * 5e1

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if rel3DCoordGT is not None:
        for i in range(numFrames):
            rel3DCoordPred = mList[i].J_transformed - mList[i].J_transformed[0:1,:]
            if isOpenGLCoord:
                rel3DCoordPred  = coordChangeMat.dot(coordChangeMat)
            loss['rel3DCoord_%d'%(i)] = (rel3DCoordPred - rel3DCoordGT[i]) * np.tile(weights[i][:,0:1], [1,3]) * 5e1


    for i in range(numFrames):
        projPts = utilsEval.chProjectPoints(mList[i].J_transformed, camMat, isOpenGLCoord)[jointsMap]
        # JVis = np.tile(np.expand_dims(JVis, 1), [1, 2])
        loss['joints2D_%d'%(i)] = (projPts - projPtsGT[i])# * np.tile(weights[i][:,0:1], [1,2])# * 1e1



    render = False

    def cbPass(_):

        pass
        # print(loss['joints'].r)



    for i in range(numFrames):
        loss['joints2D_%d' % (i)] = loss['joints2D_%d' % (i)] * np.tile(weights[i][:,0:1], [1,2]) * 1e1

    ch.minimize(loss, x0=freeVars, callback=cbPass, method='dogleg', options={'maxiter': 12})

    # vis_mesh(mList[0])



    # vis_mesh(m)
    joints3DList = []
    for i in range(numFrames):
        joints3D = mList[i].J_transformed.r[jointsMap]
        if isOpenGLCoord:
            joints3D = joints3D.dot(coordChangeMat)
        joints3DList.append(joints3D)

    # fullpose list
    fullposeList = []
    betaList = []
    transList = []
    for m in mList:
        fullposeList.append(m.fullpose.r)
        betaList.append(m.betas.r)
        transList.append(m.trans.r)

    # print(betaCh.r)
    # print((relDepPred.r - relDepGT))

    return np.stack(joints3DList, axis=0), fullposeList, betaList, transList, chGlobalPoseCoeff.r.copy(), mList












