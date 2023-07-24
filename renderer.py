import os
import sys
sys.path.append("/mnt/workplace/HOnnotate/optimization")

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

class Renderer():
    def __init__(self, device='cpu', bs=1):
        self.device = device
        self.bs = bs

    def set_renderer(self, R=None, T=None, Ext=None, intrinsics=None, image_size = (256, 256) , light_loaction = [[0.0, 0.0, 3.0]]):
        '''
        R : numpy array [3, 3]
        T : numpy array [3]
        Ext : numpy array [4, 4] or [3, 4]
        intrinsics : [3, 3]
        '''
        if R is None and T is None and Ext is None:
            M_corr = np.eye(4)
            M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
            M_obj2cam = np.matmul(M_corr, np.eye(4)[None])
        elif R is not None and T is not None and Ext is None:
            M_corr = np.eye(4)
            M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
            Ext = np.hstack([R, T[:, None]]) #[3, 4]
            Ext = np.vstack([Ext, np.array([0, 0, 0, 1])[None]]) #[4, 4]
            M_obj2cam = np.matmul(M_corr, Ext[None])
        elif R is None and T is None and Ext is not None:
            M_corr = np.eye(4)
            M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
            Ext = np.vstack([Ext, np.array([0, 0, 0, 1])[None]])
            M_obj2cam = np.matmul(M_corr, Ext[None])

        camera_R = torch.FloatTensor(np.transpose(M_obj2cam[:, :3, :3], [0, 2, 1])).repeat(self.bs, 1, 1)
        camera_T = torch.FloatTensor(M_obj2cam[:, :3, 3]).repeat(self.bs, 1)

        # Create a perspective camera
        if intrinsics is None:
            cameras = FoVPerspectiveCameras(device=self.device, fov=60.0, R=camera_R, T=camera_T)
        else:
            focal_length = (intrinsics[0, 0], intrinsics[1, 1])
            principal_point = (intrinsics[0, 2], intrinsics[1, 2])
            cameras = PerspectiveCameras(device=self.device, R=camera_R, T=camera_T, focal_length=(focal_length, ), principal_point=(principal_point, ), in_ndc=False, image_size=(image_size, ))

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=light_loaction)

        self.renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        )

        self.rasterizer_depth = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
        )
        
    def render(self, verts, faces):
        '''
        verts : [bs, V, 3]
        faces : [bs, F, 3]
        
        -> [bs, H, W, 3], [bs, H, W], [bs, H, W]
        '''
        verts_rgb = torch.ones_like(verts)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        meshes = Meshes(verts=verts, faces=faces, textures=textures)

        rgb = self.renderer_rgb(meshes)
        seg = torch.where(rgb[..., 3] != 0, 1, 0)
        depth = self.rasterizer_depth(meshes).zbuf

        return {"image":rgb[..., :3], "seg":seg, "depth":depth[..., 0]} 



