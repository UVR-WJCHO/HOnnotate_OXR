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
import transforms3d as t3d

MANO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modules/manopth/mano/models/')

class manoMesh(object):
    def __init__(self):
        self.mano_layer = ManoLayer(
            side='right', mano_root=MANO_ROOT, use_pca=False
        )

        #init param
        init_param = torch.zeros(1, 48)
        init_shape = torch.rand(1, 10)
        init_vert, init_joints = self.mano_layer(init_param, th_betas=init_shape)

        self.v = init_vert
        self.f = self.mano_layer.th_faces.repeat(init_vert.shape[0], 1, 1)
        self.j = init_joints


    def __call__(self, mano_param):
        scale = mano_param[:, 0].contiguous()    # [bs]    Scaling
        theta = mano_param[:, 1:49].contiguous()  # [bs,48] Pose parameters (mano_ncomps + rot)

        # beta = mano_param[:, 50:61].contiguous() # [bs,10] Shape parameters
        beta = torch.rand(1, 10).cuda()

        # verts, joints = self.mano_layer(
        #     torch.concat([theta, rvec], dim=1), th_betas=beta, th_trans=trans, th_scale=scale)
        verts, joints = self.mano_layer(theta, th_betas=beta)

        self.v = verts * scale
        self.j = joints * scale
        self.f = self.mano_layer.th_faces.repeat(self.v.shape[0], 1, 1) * scale


class torchRenderer():
    def __init__(self, camParam ):
        M_corr = np.eye(4)
        M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
        self.M_obj2cam = np.matmul(M_corr, np.eye(4)[None])

        # self.camera = PerspectiveCameras(focal_length=((fx, fy),), principal_point=((px, py),))


    def render(self, meshObj, K = torch.zeros(3,3), R = None, T = None, light_loaction = [[0.0, 0.0, -3.0]], image_size = 256):
        device = meshObj.v.device
        verts_rgb = torch.ones_like(meshObj.v)  # (1, V, 3)
        textrues = TexturesVertex(verts_features=verts_rgb.to(device))
        lights = PointLights(device=device, location=light_loaction)
        bs = meshObj.v.shape[0]

        if R is not None:
            if R.shape[0] != bs:
                R = R.repeat(bs, 1, 1)
        else:
            R = torch.FloatTensor(np.transpose(self.M_obj2cam[:, :3, :3], [0, 2, 1])).repeat(bs, 1, 1)

        if T is not None:
            if T.shape[0] != bs:
                T = T.repeat(bs, 1)
        else:
            T = torch.FloatTensor(self.M_obj2cam[:, :3, 3] + [[0., 0., 2.7]]).repeat(bs, 1)

        meshes = Meshes(verts=meshObj.v/100, faces=meshObj.f, textures=textrues)


        R[0, 0, 0] = 1.0
        R[0, 1, 1] = 1.0
        R_np = R.numpy()
        # cameras = PerspectiveCameras(device=device, R=R, T=T, K=K)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings_col = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        rasterizer_col = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_col
            )

        renderer_col = MeshRenderer(
            rasterizer=rasterizer_col,
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
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
        weights = torch.zeros(1, 59)    # manolayer parameter

        weights[0, 0] = 1.0     # scale
        weights[0, -10:] = torch.rand(1, 10)      # shape

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
        self.r = torchRenderer()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.1, betas=(0.5, 0.99))
        self.l1 = nn.SmoothL1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def run(self, camParam, meta, img, iter=50):
        lossperIter = []

        for i in range(iter):
            print("%d / %d"%(i+1, iter))

            losses = []
            curr_param = self.model()

            ### resources ###
            self.m(curr_param)
            hand_render = self.r.render(self.m)

            hand_color = hand_render.col
            hand_seg = hand_render.seg
            hand_depth = hand_render.depth

            color = torch.squeeze(hand_color).detach().cpu().numpy()
            depth = torch.squeeze(hand_depth).detach().cpu().numpy()
            cv2.imshow("c ", color[:, :, :-1])
            cv2.imshow("d ", depth[:, :])
            cv2.waitKey(0)

            ### losses ###
            joint_proj = self.m.j     # project mano model joints to each cam
            joint_mp = self.m.j      # mediapipe results for each cam

            loss = self.l1(joint_proj.to(self.device), torch.FloatTensor(joint_mp).to(self.device))
            losses.append(loss)

            # loss = 0
            # losses.append(loss)


            ### update ###
            totalLoss = sum(losses) / len(losses)   # check
            self.optim.zero_grad()
            totalLoss.backward()
            self.optim.step()
            lossperIter.append(totalLoss.detach().cpu())

        final_param = self.model()

        return lossperIter, final_param


if __name__ == '__main__':
    m = manoMesh()
    r = torchRenderer()
    img = r.render(m)
    color = torch.squeeze(img.col).numpy()
    cv2.imshow("r ", color[:, :, :-1])
    cv2.waitKey(0)