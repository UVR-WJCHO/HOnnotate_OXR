import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras
)

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import transforms3d as t3d
from HOnnotate_refine.eval.utilsEval import showHandJoints

import chumpy as ch
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model as load_mano_model
import pickle


MANO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modules/manopth/mano/models/')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

h = np.array([[0,0,0,1]])
jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]


def mano_param_dict(n_pose_params, n_betas=10):
    out = {
        'pose': [0.0 for _ in range(n_pose_params + 3)],
        'betas': [0.0 for _ in range(n_betas)],
        'valid': False,
        'mTc': {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [1.0, 0.0, 0.0, 0.0],
        }
    }
    return out

def cv2ProjectPoints(camMat, pts3D, isOpenGLCoords=True):
    '''
    TF function for projecting 3d points to 2d using CV2
    :param camProp:
    :param pts3D:
    :param isOpenGLCoords:
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts

def torchProjectPoints(camMat, pts3D, isOpenGLCoords=False, isPytorch3DCoords=False):
    '''
    TF function for projecting 3d points to 2d using CV2
    :param camProp:
    :param pts3D:
    :param isOpenGLCoords:
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = torch.tensor([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=torch.float32).cuda()
    coordChangeMat_py3D = torch.tensor([[-1., 0., 0.], [0, -1., 0.], [0., 0., 1.]], dtype=torch.float32).cuda()
    if isOpenGLCoords:
        pts3D = pts3D @ coordChangeMat.T
    if isPytorch3DCoords:
        pts3D = pts3D @ coordChangeMat_py3D.T

    projPts = pts3D @ camMat.T
    projPts = torch.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts


class manoMesh(object):
    def __init__(self):
        self.mano_layer = ManoLayer(
            side='right', mano_root=MANO_ROOT, use_pca=False, flat_hand_mean=False, robust_rot=True
        )

        #init param
        init_param = torch.zeros(1, 48)
        init_vert, init_joints = self.mano_layer(init_param)

        self.v = init_vert
        self.f = self.mano_layer.th_faces.repeat(init_vert.shape[0], 1, 1)
        self.j = init_joints


    # def __call__(self, mano_param):
    #     scale = mano_param[:, 0].contiguous()    # [bs]    Scaling
    #     theta = mano_param[:, 1:49].contiguous()  # [bs,48] Pose parameters (mano_ncomps + rot)
    #     beta = mano_param[:, 49:59].contiguous() # [bs,10] Shape parameters
    #     trans = mano_param[:, 59:].contiguous()  # [bs,2] Shape parameters
    #
    #     # verts, joints = self.mano_layer(
    #     #     torch.concat([theta, rvec], dim=1), th_betas=beta, th_trans=trans, th_scale=scale)
    #     verts, joints = self.mano_layer(theta, th_betas=beta, th_trans=trans)
    #
    #     self.v = verts #* scale
    #     self.j = joints #* scale
    #
    #     self.face = self.mano_layer.th_faces
    #     self.f = self.mano_layer.th_faces.repeat(self.v.shape[0], 1, 1) #* scale

    def __call__(self, mano_param):
        self.scale = mano_param[:, 0].contiguous()  # [bs]    Scaling
        self.trans = mano_param[:, 1:3].contiguous()  # [bs,2]  Translation in x and y only
        self.rvec = mano_param[:, 3:6].contiguous()  # [bs,3]  Global rotation vector
        self.beta = mano_param[:, 6:16].contiguous()  # [bs,10] Shape parameters
        self.theta = mano_param[:, 16:].contiguous()  # [bs,45] Angle parameters

        verts, joints = self.mano_layer(
            torch.concat([self.theta, self.rvec], dim=1), th_betas=self.beta, th_trans=self.trans)

        self.v = verts * self.scale
        self.f = self.mano_layer.th_faces.repeat(self.v.shape[0], 1, 1)
        self.j = joints * self.scale

class torchRenderer():
    def __init__(self, device):
        self.device = device

        self.K = torch.unsqueeze(torch.eye(4), 0)
        self.Rot = torch.unsqueeze(torch.eye(3), 0)
        self.Trans = torch.unsqueeze(torch.zeros(3), 0)


            self.Trans[:, 0] = -0.8
            self.Trans[:, 1] = -0.8
            self.Trans[:, 2] = 1.0
            self.camera = PerspectiveCameras(device=self.device, R=self.Rot, T=self.Trans, K=self.K)




    def setCam(self, K, R, T):
        self.camera = PerspectiveCameras(device=self.device, R=R, T=T, K=K)

    def render(self, meshObj, light_loaction = [[0.0, 0.0, -3.0]], image_size = [256, 256], R = None, T = None):
        verts_rgb = torch.ones_like(meshObj.v)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        lights = PointLights(device=self.device, location=light_loaction)
        meshes = Meshes(verts=meshObj.v/100, faces=meshObj.f, textures=textures)

        # cameras = PerspectiveCameras(device=device, R=R, T=T, K=K)
        raster_settings_col = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        rasterizer_col = MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings_col
            )

        renderer_col = MeshRenderer(
            rasterizer=rasterizer_col,
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=lights,
            )
        )

        images_col = renderer_col(meshes)
        images_seg = torch.where(images_col[0, ..., 3] != 0, 1, 0)
        depth_col = rasterizer_col(meshes).zbuf

        class imgObject(object):
            pass

        imgObject.col = images_col
        imgObject.seg = images_seg
        imgObject.depth = depth_col

        return imgObject


class handModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.zeros(1, 61)    # manolayer parameter

        initWeight = torch.tensor([[ 1.4858e+00,  0.0000e+00,  0.0000e+00, -1.6937e-01, -2.2858e-01,
         -1.2013e+00,  1.3280e+00, -1.3427e+00, -3.2712e+00, -1.0803e+00,
          6.1840e+00,  2.6016e-01,  3.2001e+00,  3.1131e+00,  1.8295e+00,
         -3.1644e+00, -1.3311e+00,  3.8767e-01, -1.0578e+00,  1.3989e+00,
          9.2507e-02, -6.5948e-01,  1.2553e+00,  4.7081e-01, -1.3085e+00,
          4.7761e-01,  6.7092e-01, -5.1978e-01,  1.0701e+00,  5.1424e-01,
         -3.4349e-01,  1.4245e+00,  3.6803e-01, -6.5561e-01,  2.0262e+00,
          1.4157e+00, -2.2327e+00,  1.8233e+00,  8.3574e-01,  3.3260e-01,
          1.0889e-01, -4.9964e-01, -1.6107e-01,  1.1358e+00, -2.7578e-01,
          7.9584e-02,  2.2094e+00,  5.7550e-01, -2.1367e-01,  6.9795e-01,
         -2.0888e-01, -4.2746e-01, -1.1274e+00,  1.1778e+00,  9.7144e-01,
          3.4753e+00, -1.1552e+00, -7.3388e-01, -7.2311e-01, -4.1669e-01,
         -2.4556e-03]])

        weights = initWeight


        weights[0, 0] = 1.0     # scale
        weights[0, 6:16] = torch.rand(1, 10)      # shape

        weights.requires_grad = True
        self.weights = nn.Parameter(weights)

    def forward(self):
        return self.weights


class jointModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.zeros(1, 21, 3)    # manolayer parameter

        weights.requires_grad = True
        self.weights = nn.Parameter(weights)

    def forward(self):
        return self.weights


class optimizer_torch():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % 0 if use_cuda else "cpu")

        self.m = manoMesh()
        self.m.mano_layer.to(self.device)

        self.model = handModel().to(self.device)       # model = nn.DataParallel(model)
        self.joint = jointModel().to(self.device)

        self.r = torchRenderer(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.1, betas=(0.5, 0.99))
        self.l1 = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def run(self, camSet, metas, rgbSet, depthSet, iter=50):
        lossperIter = []


        a = torch.ones((21, 1)).cuda()
        h = torch.tensor([[0, 0, 0, 1]]).cuda()

        manualDepth = torch.ones((21, 1)).cuda()
        manualDepth *= -20.0

        for i in range(iter):
            print("%d / %d"%(i+1, iter))
            losses = []
            curr_param = self.model()
            self.m(curr_param)

            ## render mesh
            hand_render = self.r.render(self.m)
            hand_color = hand_render.col
            hand_depth = hand_render.depth

            color = torch.squeeze(hand_color).detach().cpu().numpy()
            depth = torch.squeeze(hand_depth).detach().cpu().numpy()
            cv2.imshow("c ", color[:, :, :-1])
            cv2.imshow("d ", depth[:, :])
            cv2.waitKey(0)

            ### 1. transform mano space pts to world space with renderer camera parameter.
            # R = torch.squeeze(self.r.Rot)
            # T = self.r.Trans.T
            # ext = torch.cat((R, T), axis=-1)
            # extrinsic = torch.FloatTensor(ext).cuda()
            #
            # mano3Dcam = torch.squeeze(self.m.j)
            # mano4Dcam = torch.concatenate([mano3Dcam, a], axis=1)
            # projMat = torch.concatenate((extrinsic, h), 0)
            # inv = torch.linalg.inv(projMat)
            # mano4Dworld = (inv @ mano4Dcam.T).T

            ### or 2. mano space is world space(w. default camera parameter)
            mano3Dworld = torch.squeeze(self.m.j)
            mano4Dworld = torch.concatenate([mano3Dworld, a], axis=1)

            ### or 3. transform to world space with main cam
            # intrinsic, extrinsic = camSet[0]
            # intrinsic = torch.FloatTensor(intrinsic).cuda()
            # extrinsic = torch.FloatTensor(extrinsic).cuda()
            #
            # mano3Dcam = torch.squeeze(self.m.j)
            # mano4Dcam = torch.concatenate([mano3Dcam, a], axis=1)
            # projMat = torch.concatenate((extrinsic, h), 0)
            #
            # mano4Dworld = (torch.linalg.inv(projMat) @ mano4Dcam.T).T

            for idx, camParam in enumerate(camSet):
                ### get camera parameter for each cam
                meta = metas[idx]
                intrinsic, extrinsic = camParam
                intrinsic = torch.FloatTensor(intrinsic).cuda()
                extrinsic = torch.FloatTensor(extrinsic).cuda()
                img2bb = torch.FloatTensor(meta['img2bb']).cuda()

                ### transform world coordinate pts to each camera coordinate
                mano4Dtargetcam = (extrinsic @ mano4Dworld.T).T
                projPts = torchProjectPoints(intrinsic, mano4Dtargetcam, isPytorch3DCoords=True)#[jointsMap]

                ### transform camera coordinate pts to cropped region
                uv1 = torch.concatenate((projPts, torch.ones_like(projPts[:, :1])), 1)
                projPts = (img2bb @ uv1.T).T

                # get GT 2D keypoints(mediapipe)
                projPtsGT = meta['kpts_crop'][:, 0:2]
                if np.isnan(projPtsGT[0, 0]):
                    continue
                projPtsGT = torch.FloatTensor(projPtsGT).cuda()

                # compute 2D joint loss per camera
                loss = self.l1(projPtsGT.to(self.device), projPts.to(self.device))
                losses.append(loss)

                # # compute depth loss (manual depth)
                loss = self.l1(mano4Dtargetcam[:, -1], manualDepth)
                losses.append(loss)
                break

            ### update ###
            totalLoss = sum(losses) / len(losses)
            print(totalLoss)

            self.optim.zero_grad()
            totalLoss.backward()
            self.optim.step()
            lossperIter.append(totalLoss.detach().cpu())

            # camMas = camSet[0]
            # intrinsic, extrinsic = camMas
            # K = np.eye(4, dtype=float)
            # K[:3, :3] = intrinsic
            # R = extrinsic[:, :-1]
            # T = extrinsic[:, -1]
            #
            # K = torch.unsqueeze(torch.FloatTensor(K), 0)
            # R = torch.unsqueeze(torch.FloatTensor(R), 0)
            # T = torch.unsqueeze(torch.FloatTensor(T), 0)
            #
            # self.r.setCam(K, R, T)

            curr_param = self.model()
            self.m(curr_param)


            rgbMas = cv2.imread(rgbSet[0])
            height = rgbMas.shape[0]
            width = rgbMas.shape[1]

            projPtsGT = metas[0]['kpts_crop'][:, 0:2]
            rgbGT = showHandJoints(rgbMas.copy(), np.copy(projPtsGT).astype(np.float32), estIn=None,
                                                 filename=None,
                                                 upscale=1, lineThickness=3)

            projPts = np.squeeze(projPts.detach().cpu().numpy())[:, 0:2]
            rgbPred = showHandJoints(rgbMas.copy(), np.copy(projPts).astype(np.float32), estIn=None,
                                       filename=None,
                                       upscale=1, lineThickness=3)

            # cv2.imshow("input", rgbMas)
            cv2.imshow("GT", rgbGT)
            cv2.imshow("Pred", rgbPred)

            ## render mesh
            # hand_render = self.r.render(self.m)
            # hand_color = hand_render.col
            # hand_depth = hand_render.depth
            #
            # color = torch.squeeze(hand_color).detach().cpu().numpy() * 255.0
            # depth = torch.squeeze(hand_depth).detach().cpu().numpy() * 255.0
            # cv2.imshow("c ", color[:, :, :-1])
            # cv2.imshow("d ", depth[:, :])
            #
            cv2.waitKey(1)


        ### note
        # mano model joint 0 looks like anchord in 0, 0

        final_param = self.model()

        return lossperIter, final_param

    def debug(self, camSet, metas, rgbSet, depthSet, iter=50):
        lossperIter = []

        a = torch.ones((21, 1)).cuda()
        h = torch.tensor([[0, 0, 0, 1]]).cuda()

        manualDepth = torch.ones((21, 1)).cuda()
        manualDepth *= -30.0

        manualWrist = torch.ones((2, 1)).cuda()
        manualWrist *= 50.0

        for i in range(iter):
            print("%d / %d"%(i+1, iter))
            losses = []
            curr_param = self.model()
            self.m(curr_param)

            ## render mesh
            hand_render = self.r.render(self.m)
            hand_color = hand_render.col
            hand_depth = hand_render.depth

            color = torch.squeeze(hand_color).detach().cpu().numpy()
            depth = torch.squeeze(hand_depth).detach().cpu().numpy()
            cv2.imshow("c ", color[:, :, :-1])
            cv2.imshow("d ", depth[:, :])
            cv2.waitKey(0)

            ### 1. transform mano space pts to world space with renderer camera parameter.
            # R = torch.squeeze(self.r.Rot)
            # T = self.r.Trans.T
            # ext = torch.cat((R, T), axis=-1)
            # extrinsic = torch.FloatTensor(ext).cuda()
            #
            # mano3Dcam = torch.squeeze(self.m.j)
            # mano4Dcam = torch.concatenate([mano3Dcam, a], axis=1)
            # projMat = torch.concatenate((extrinsic, h), 0)
            # inv = torch.linalg.inv(projMat)
            # mano4Dworld = (inv @ mano4Dcam.T).T

            ### or 2. mano space is world space(w. default camera parameter)
            # mano3Dworld = torch.squeeze(self.m.j)
            # mano4Dworld = torch.concatenate([mano3Dworld, a], axis=1)

            ### or 3. transform to world space with main cam
            intrinsic, extrinsic = camSet[0]
            intrinsic = torch.FloatTensor(intrinsic).cuda()
            extrinsic = torch.FloatTensor(extrinsic).cuda()

            mano3Dcam = torch.squeeze(self.m.j)
            mano4Dcam = torch.concatenate([mano3Dcam, a], axis=1)
            projMat = torch.concatenate((extrinsic, h), 0)

            mano4Dworld = (torch.linalg.inv(projMat) @ mano4Dcam.T).T

            kps3D = mano4Dworld[:, :3]

            coordChangeMat = torch.tensor([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=torch.float32).cuda()
            coordChangeMat_py3D = torch.tensor([[-1., 0., 0.], [0, -1., 0.], [0., 0., 1.]], dtype=torch.float32).cuda()
            kps3D = kps3D @ coordChangeMat_py3D.T

            relkps3D = kps3D - kps3D[0, :]

            projPtsGT = np.copy(metas[0]['kpts_crop'][:, 0:2])
            projPtsGT = projPtsGT - projPtsGT[0, :]


            projPtsGT = torch.FloatTensor(projPtsGT).cuda()

            # compute relative 2D joint loss per camera
            loss = self.mse(projPtsGT[1:, :].to(self.device), relkps3D[1:, :2].to(self.device))
            losses.append(loss)
            # compute  2D joint loss per camera
            # loss = self.mse(projPtsGT.to(self.device), kps3D[:, :2].to(self.device))
            # losses.append(loss)

            loss = self.mse(manualDepth.to(self.device), kps3D[:, 2].to(self.device))
            losses.append(loss)
            #
            # loss = self.mse(projPtsGT[0, :], kps3D[0, :2]) * 10.0
            # losses.append(loss)

            ### update ###
            totalLoss = sum(losses) / len(losses)
            print(totalLoss)

            self.optim.zero_grad()
            totalLoss.backward()
            self.optim.step()

            lossperIter.append(totalLoss.detach().cpu())

            # camMas = camSet[0]
            # intrinsic, extrinsic = camMas
            # K = np.eye(4, dtype=float)
            # K[:3, :3] = intrinsic
            # R = extrinsic[:, :-1]
            # T = extrinsic[:, -1]
            #
            # K = torch.unsqueeze(torch.FloatTensor(K), 0)
            # R = torch.unsqueeze(torch.FloatTensor(R), 0)
            # T = torch.unsqueeze(torch.FloatTensor(T), 0)
            #
            # self.r.setCam(K, R, T)

            curr_param = self.model()
            self.m(curr_param)


            rgbMas = cv2.imread(rgbSet[0])
            height = rgbMas.shape[0]
            width = rgbMas.shape[1]

            # projPtsGT = metas[0]['kpts_crop'][:, 0:2]
            # rgbGT = showHandJoints(rgbMas.copy(), np.copy(projPtsGT).astype(np.float32), estIn=None,
            #                                      filename=None,
            #                                      upscale=1, lineThickness=3)
            #
            # projPts = np.squeeze(projPts.detach().cpu().numpy())[:, 0:2]
            # rgbPred = showHandJoints(rgbMas.copy(), np.copy(projPts).astype(np.float32), estIn=None,
            #                            filename=None,
            #                            upscale=1, lineThickness=3)
            #
            # cv2.imshow("input", rgbMas)
            # cv2.imshow("GT", rgbGT)
            # cv2.imshow("Pred", rgbPred)

            ## render mesh
            # hand_render = self.r.render(self.m)
            # hand_color = hand_render.col
            # hand_depth = hand_render.depth
            #
            # color = torch.squeeze(hand_color).detach().cpu().numpy() * 255.0
            # depth = torch.squeeze(hand_depth).detach().cpu().numpy() * 255.0
            # cv2.imshow("c ", color[:, :, :-1])
            # cv2.imshow("d ", depth[:, :])
            #
            cv2.waitKey(1)


        ### note
        # mano model joint 0 looks like anchord in 0, 0

        final_param = self.model()

        return lossperIter, final_param


    def debug2(self, camSet, metas, rgbSet, depthSet, iter=50):
        lossperIter = []

        for idx, camParam in enumerate(camSet):
            meta = metas[idx]
            projPtsGT = meta['kpts_crop'][:, 0:2]

        for i in range(iter):
            print("%d / %d"%(i+1, iter))
            losses = []
            curr_param = self.model()
            self.m(curr_param)

            ## render mesh
            hand_render = self.r.render(self.m)
            hand_color = hand_render.col
            hand_depth = hand_render.depth

            color = torch.squeeze(hand_color).detach().cpu().numpy()
            depth = torch.squeeze(hand_depth).detach().cpu().numpy()
            cv2.imshow("c ", color[:, :, :-1])
            cv2.imshow("d ", depth[:, :])
            cv2.waitKey(0)

            ### 1. transform mano space pts to world space with renderer camera parameter.
            # R = torch.squeeze(self.r.Rot)
            # T = self.r.Trans.T
            # ext = torch.cat((R, T), axis=-1)
            # extrinsic = torch.FloatTensor(ext).cuda()
            #
            # mano3Dcam = torch.squeeze(self.m.j)
            # mano4Dcam = torch.concatenate([mano3Dcam, a], axis=1)
            # projMat = torch.concatenate((extrinsic, h), 0)
            # inv = torch.linalg.inv(projMat)
            # mano4Dworld = (inv @ mano4Dcam.T).T

            ### or 2. mano space is world space(w. default camera parameter)
            mano3Dworld = torch.squeeze(self.m.j)
            mano4Dworld = torch.concatenate([mano3Dworld, a], axis=1)

            ### or 3. transform to world space with main cam
            # intrinsic, extrinsic = camSet[0]
            # intrinsic = torch.FloatTensor(intrinsic).cuda()
            # extrinsic = torch.FloatTensor(extrinsic).cuda()
            #
            # mano3Dcam = torch.squeeze(self.m.j)
            # mano4Dcam = torch.concatenate([mano3Dcam, a], axis=1)
            # projMat = torch.concatenate((extrinsic, h), 0)
            #
            # mano4Dworld = (torch.linalg.inv(projMat) @ mano4Dcam.T).T

            for idx, camParam in enumerate(camSet):
                ### get camera parameter for each cam
                meta = metas[idx]
                intrinsic, extrinsic = camParam
                intrinsic = torch.FloatTensor(intrinsic).cuda()
                extrinsic = torch.FloatTensor(extrinsic).cuda()
                img2bb = torch.FloatTensor(meta['img2bb']).cuda()

                ### transform world coordinate pts to each camera coordinate
                mano4Dtargetcam = (extrinsic @ mano4Dworld.T).T
                projPts = torchProjectPoints(intrinsic, mano4Dtargetcam, isPytorch3DCoords=True)#[jointsMap]

                ### transform camera coordinate pts to cropped region
                uv1 = torch.concatenate((projPts, torch.ones_like(projPts[:, :1])), 1)
                projPts = (img2bb @ uv1.T).T

                # get GT 2D keypoints(mediapipe)
                projPtsGT = meta['kpts_crop'][:, 0:2]
                if np.isnan(projPtsGT[0, 0]):
                    continue
                projPtsGT = torch.FloatTensor(projPtsGT).cuda()

                # compute 2D joint loss per camera
                loss = self.l1(projPtsGT.to(self.device), projPts.to(self.device))
                losses.append(loss)

                # # compute depth loss (manual depth)
                loss = self.l1(mano4Dtargetcam[:, -1], manualDepth)
                losses.append(loss)
                break

            ### update ###
            totalLoss = sum(losses) / len(losses)
            print(totalLoss)

            self.optim.zero_grad()
            totalLoss.backward()
            self.optim.step()
            lossperIter.append(totalLoss.detach().cpu())

            # camMas = camSet[0]
            # intrinsic, extrinsic = camMas
            # K = np.eye(4, dtype=float)
            # K[:3, :3] = intrinsic
            # R = extrinsic[:, :-1]
            # T = extrinsic[:, -1]
            #
            # K = torch.unsqueeze(torch.FloatTensor(K), 0)
            # R = torch.unsqueeze(torch.FloatTensor(R), 0)
            # T = torch.unsqueeze(torch.FloatTensor(T), 0)
            #
            # self.r.setCam(K, R, T)

            curr_param = self.model()
            self.m(curr_param)


            rgbMas = cv2.imread(rgbSet[0])
            height = rgbMas.shape[0]
            width = rgbMas.shape[1]

            projPtsGT = metas[0]['kpts_crop'][:, 0:2]
            rgbGT = showHandJoints(rgbMas.copy(), np.copy(projPtsGT).astype(np.float32), estIn=None,
                                                 filename=None,
                                                 upscale=1, lineThickness=3)

            projPts = np.squeeze(projPts.detach().cpu().numpy())[:, 0:2]
            rgbPred = showHandJoints(rgbMas.copy(), np.copy(projPts).astype(np.float32), estIn=None,
                                       filename=None,
                                       upscale=1, lineThickness=3)

            # cv2.imshow("input", rgbMas)
            cv2.imshow("GT", rgbGT)
            cv2.imshow("Pred", rgbPred)

            ## render mesh
            # hand_render = self.r.render(self.m)
            # hand_color = hand_render.col
            # hand_depth = hand_render.depth
            #
            # color = torch.squeeze(hand_color).detach().cpu().numpy() * 255.0
            # depth = torch.squeeze(hand_depth).detach().cpu().numpy() * 255.0
            # cv2.imshow("c ", color[:, :, :-1])
            # cv2.imshow("d ", depth[:, :])
            #
            cv2.waitKey(1)


        ### note
        # mano model joint 0 looks like anchord in 0, 0

        final_param = self.model()

        return lossperIter, final_param



if __name__ == '__main__':
    m = manoMesh()
    r = torchRenderer()
    img = r.render(m)
    color = torch.squeeze(img.col).numpy()
    cv2.imshow("r ", color[:, :, :-1])
    cv2.waitKey(0)