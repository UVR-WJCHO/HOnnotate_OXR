import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), '../HOnnotate_refine/optimization'))

import numpy as np
import torch
import torch.nn as nn
import cv2

from torch.nn import functional as F
from manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    BlendParams,
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
from scipy.spatial.transform import Rotation as R
import time


def changeCoordtopytorch3D(extrinsic):
    extrinsic_py = np.copy(extrinsic)

    # axis flip rotation value
    r = R.from_euler('z', [180], degrees=True)
    r = np.squeeze(r.as_matrix())
    extrinsic_py = r @ extrinsic_py

    return extrinsic_py


class Renderer(nn.Module):
    def __init__(self, device='cuda', bs=1, extrinsic=None, intrinsic=None, image_size=None, light_loaction=((2.0, 2.0, -2.0),)):
        super().__init__()
        '''
            R : numpy array [3, 3]
            T : numpy array [3]
            Ext : numpy array [4, 4] or [3, 4]
            intrinsics : [3, 3]
        '''
        self.device = device
        self.bs = bs

        ### HTW version
        """
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
        """

        ### WJ version
        focal_l = (intrinsic[0, 0], intrinsic[1, 1])
        principal_p = (intrinsic[0, -1], intrinsic[1, -1])

        extrinsic_py3D = changeCoordtopytorch3D(extrinsic)

        R = torch.unsqueeze(torch.FloatTensor(extrinsic_py3D[:, :-1]), 0)
        T = torch.unsqueeze(torch.FloatTensor(extrinsic_py3D[:, -1]), 0)

        cameras = PerspectiveCameras(device=self.device, image_size=(image_size,), focal_length=(focal_l,),
                                     principal_point=(principal_p,), R=R, T=T, in_ndc=False)

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size = None,
            max_faces_per_bin = None
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

        ## require 0.7 sec to render
        # blend_params = BlendParams(sigma=1e-7, gamma=1e-7)
        # self.silhouette_renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=cameras,
        #         raster_settings=raster_settings
        #     ),
        #     shader=SoftSilhouetteShader(blend_params=blend_params)
        # )


    def render_simple(self, meshes, flag_rgb=False):
        '''
        verts : [bs, V, 3]
        faces : [bs, F, 3]

        -> [bs, H, W, 3], [bs, H, W], [bs, H, W]
        '''

        if flag_rgb:
            rgb = self.renderer_rgb(meshes)[..., :3]
        else:
            rgb = None
        t1 = time.time()
        depth = self.rasterizer_depth(meshes).zbuf[..., 0]

        print("raster d : ", time.time() - t1)

        # depth map process
        depth[depth == -1] = 0.
        depth = depth * 10.0    # change to mm scale (same as gt)

        seg = torch.empty_like(depth).copy_(depth)

        # loss_depth = torch.sum(((depth_rendered - self.depth_ref / self.scale) ** 2).view(self.batch_size, -1),
        #                        -1) * 0.00012498664727900177  # depth scale used in HOnnotate

        return {"rgb":rgb, "depth":depth, "seg":seg}


    def compute_depth_loss(self, pred_depth):
        # pred_depth_np = np.squeeze(pred_depth.cpu().detach().numpy())
        # pred_depth_vis = (pred_depth_np * 100).astype(np.uint8)
        # cv2.imshow("pred_depth_vis", pred_depth_vis)
        # depth_ref_np = np.squeeze(self.depth_ref.cpu().detach().numpy())
        # depth_ref_vis = (depth_ref_np * 100).astype(np.uint8)
        # cv2.imshow("depth_ref_vis", depth_ref_vis)
        # cv2.waitKey(0)

        depth_gap = torch.abs(pred_depth - self.depth_ref)

        depth_loss = torch.sum(depth_gap) / (640*480)

        depth_gap_np = np.squeeze(depth_gap.cpu().detach().numpy())

        return depth_loss, depth_gap

    def compute_seg_loss(self, pred_seg):
        seg_a = self.seg_ref - pred_seg
        seg_a[seg_a < 0] = 0.0
        seg_b = pred_seg - self.seg_ref
        seg_b[seg_b < 0] = 0.0

        seg_gap = seg_a + seg_b
        # seg_gap[self.seg_ref == pred_seg] = 0
        # seg_gap[self.seg_ref == 10] = 0
        # seg_gap[pred_seg == 10] = 0

        seg_loss = torch.sum(seg_gap)
        return seg_loss, seg_gap


    def register_seg(self, seg_ref):
        self.register_buffer('seg_ref', seg_ref)

    def register_depth(self, depth_ref):
        self.register_buffer('depth_ref', depth_ref)

    def render(self, verts, faces, flag_rgb=False):
        '''
        verts : [bs, V, 3]
        faces : [bs, F, 3]
        
        -> [bs, H, W, 3], [bs, H, W], [bs, H, W]
        '''
        verts_rgb = torch.ones_like(verts, device=self.device)
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = Meshes(verts=verts, faces=faces, textures=textures)

        if flag_rgb:
            rgb = self.renderer_rgb(meshes)[..., :3]
        else:
            rgb = None

        depth = self.rasterizer_depth(meshes).zbuf[..., 0]
        # depth map process
        depth[depth == -1] = 0.
        seg = torch.empty_like(depth).copy_(depth)

        # depth = depth * 10.0    # change to mm scale (same as gt)
        depth = depth / 100.0   # change to m scale
        depth[depth == 0] = 10.


        # loss_depth = torch.sum(((depth_rendered - self.depth_ref / self.scale) ** 2).view(self.batch_size, -1),
        #                        -1) * 0.00012498664727900177  # depth scale used in HOnnotate

        self.output = {"rgb":rgb, "depth":depth, "seg":seg}

        return self.output

    def render_meshes(self, verts_list, faces_list, flag_rgb=False):
        '''
        verts : [bs, V, 3]
        faces : [bs, F, 3]

        -> [bs, H, W, 3], [bs, H, W], [bs, H, W]
        '''
        mesh_list = list()
        for verts, faces in zip(verts_list, faces_list):
            verts_rgb = torch.ones_like(verts, device=self.device)
            textures = TexturesVertex(verts_features=verts_rgb)
            meshes = Meshes(verts=verts, faces=faces, textures=textures)
            mesh_list.append(meshes)

        mesh_joined = join_meshes_as_scene(mesh_list)

        if flag_rgb:
            rgb = self.renderer_rgb(mesh_joined)[..., :3]
        else:
            rgb = None

        depth = self.rasterizer_depth(mesh_joined).zbuf

        # depth map process
        depth[depth == -1] = 0.
        depth = depth * 10.0

        seg = torch.empty_like(depth).copy_(depth)

        # loss_depth = torch.sum(((depth_rendered - self.depth_ref / self.scale) ** 2).view(self.batch_size, -1),
        #                        -1) * 0.00012498664727900177  # depth scale used in HOnnotate

        return {"rgb": rgb, "depth": depth[..., 0], "seg":seg[..., 0]}

