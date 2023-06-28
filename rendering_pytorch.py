import os
import sys
sys.path.append("/mnt/workplace/HOnnotate/optimization")
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
        init_vert, init_joints = self.mano_layer(
            init_param
        )
        self.v = init_vert
        self.f = self.mano_layer.th_faces.repeat(init_vert.shape[0], 1, 1)
        self.j = init_joints

    def __call__(self, mano_param):
        scale = mano_param[:, 0].contiguous()    # [bs]    Scaling 
        trans = mano_param[:, 1:3].contiguous()  # [bs,2]  Translation in x and y only
        rvec  = mano_param[:, 3:6].contiguous()  # [bs,3]  Global rotation vector
        beta  = mano_param[:, 6:16].contiguous() # [bs,10] Shape parameters
        theta   = mano_param[:, 16:].contiguous()  # [bs,45] Angle parameters

        verts, joints = self.mano_layer(
            torch.concat([theta, rvec], dim=1), th_betas=beta, th_trans = trans, th_scale = scale
        )
        
        self.v = verts
        self.j = joints


class TorchRenderer():
    def __init__(self):
        M_corr = np.eye(4)
        M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
        self.M_obj2cam = np.matmul(M_corr, np.eye(4)[None])


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


if __name__ == '__main__':
    print(".")

    m = manoMesh()
    r = TorchRenderer()
    img = r.render(m)
    color = torch.squeeze(img.col).numpy()
    cv2.imshow("r ", color[:, :, :-1])
    cv2.waitKey(0)