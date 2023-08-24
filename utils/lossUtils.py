import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import cv2
import numpy as np
import math
from utils import params

from pytorch3d.structures import Meshes, Pointclouds

from utils.lossUtils import *
from utils.modelUtils import *

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable

def mano3DToCam3D(xyz3D, ext, main_ext):
    device = xyz3D.device
    h = torch.tensor([[0, 0, 0, 1]]).to(device)

    xyz3D = torch.squeeze(xyz3D)
    ones = torch.ones((xyz3D.shape[0], 1)).to(device)

    # mano to world
    xyz4Dcam = torch.cat([xyz3D, ones], axis=1)
    projMat = torch.cat((main_ext, h), 0)
    xyz4Dworld = (torch.linalg.inv(projMat) @ xyz4Dcam.T).T

    # world to target cam
    xyz3Dcam2 = xyz4Dworld @ ext.T  # [:, :3]

    return xyz3Dcam2

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    if xyz.shape[0] != K.shape[0]:
        K = K.repeat(xyz.shape[0], 1, 1).to(xyz.device)
    uv = torch.matmul(K, xyz.transpose(2, 1)).transpose(2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]

def get_pose_constraint_tensor():
    pose_mean_tensor = torch.tensor(params.pose_mean_list).cuda()
    pose_reg_tensor = torch.tensor(params.pose_reg_list).cuda()
    return pose_reg_tensor, pose_mean_tensor

def paint_kpts(img_path, img, kpts, circle_size = 1):
    colors = params.colors
    # To be continued
    limbSeq = params.limbSeq_hand

    # convert axis
    kpts = kpts[:, [1, 0]]
    
    im = cv2.imread(img_path) if img is None else img.copy()
    # draw points
    for k, kpt in enumerate(kpts):
        row = int(kpt[0])
        col = int(kpt[1])
        if k in [0, 4, 8, 12, 16, 20]:
            r = circle_size
        else:
            r = 1
        cv2.circle(im, (col, row), radius=r, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [X0, Y0] = kpts[limb[0]]
        [X1, Y1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

    return im


_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):

        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


def point_mesh_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area)

    return point_to_face

