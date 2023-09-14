def mano3DToCam3D(xyz3D, ext):
    device = xyz3D.device

    xyz3D = torch.squeeze(xyz3D)
    ones = torch.ones((xyz3D.shape[0], 1), device=device)

    # mano to world
    xyz4Dcam = torch.cat([xyz3D, ones], axis=1)
    # projMat = torch.concatenate((main_ext, h), 0)
    # xyz4Dworld = (torch.linalg.inv(projMat) @ xyz4Dcam.T).T

    # world to target cam
    xyz3Dcam2 = xyz4Dcam @ ext.T  # [:, :3]

    return xyz3Dcam2


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    if xyz.shape[0] != K.shape[0]:
        K = K.repeat(xyz.shape[0], 1, 1).to(xyz.device)
    uv = torch.matmul(K, xyz.transpose(2, 1)).transpose(2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]


# Ms : camera extrinsic
# Ks : camera intrinsic


joints_cam = torch.unsqueeze(mano3DToCam3D({'annotation-data-xyz'}, Ms[camIdx]), 0)
pred_kpts2d = projectPoints(joints_cam, Ks[camIdx])