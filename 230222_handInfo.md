[visualize code sample for kps3D in .pickle]
    ### required info
    # mainCampose(extrinsic) : main camera during pose optimization(fixed, currently sub1)
    # camPose(extrinsic) : (3,4) extrinsic parameter of target camera for visualize
    # camMat(intrinsic) : (3,3) intrinsic parameter of target camera for visualize
    # CamID : target camera ID, index of camIDset = ['mas', 'sub1', 'sub2', 'sub3']
    
from HOnnotate_refine.eval import utilsEval
    
    kps3D = sample['KPS3D']
    scale = sample['scale']
    meta = sample['meta']
    
    img2bb = meta[CamID]['img2bb']
    
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
    imgOutEst = utilsEval.showHandJoints(img.copy(), np.copy(projPts).astype(np.float32), estIn=None, filename=None, upscale=1, lineThickness=3)
    axEst.imshow(imgOutEst[:, :, [2, 1, 0]])
    
