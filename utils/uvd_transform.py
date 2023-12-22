import numpy as np
import sys
import torch as torch
# import chumpy as np
sys.path.append('.')
sys.path.append('..')

def uvdtoxyz(uvd, K):
    device = uvd.device
    depth = uvd[:, 2] #[21]
    depth = torch.reshape(depth, [21,1]) #[21,1]
    # depth = torch.tile(depth, [1,3]) #[21,3]
    depth = depth.repeat(1,3)
    one = torch.ones(21).reshape(21,1).to(device)
    # uvd_nodepth = torch.concat((uvd[:,:-1], one), axis=1)
    uvd_nodepth = torch.cat((uvd[:,:-1], one), axis=1)
    uvd_scaled = uvd_nodepth * depth
    # xyz = torch.linalg.matmul(torch.linalg.inv(K), uvd_scaled.T) #[3, 21]
    xyz = torch.matmul(torch.inverse(K), uvd_scaled.T) #[3, 21]

    return xyz.T #[21, 3]

def xyztouvd(xyz, K):
    device = xyz.device
    # uvd = torch.linalg.matmul(K, xyz.T).T
    uvd = torch.matmul(K, xyz.T).T
    scale = uvd[:, 2]
    one = torch.ones(21).reshape(21,1).to(device)
    # depth = torch.concat((one, one, torch.reshape(scale, [21,1])), axis=1)
    depth = torch.cat((one, one, torch.reshape(scale, [21,1])), axis=1)
    # one = np.ones()
    # scale_expand = np.tile(scale, [3,1]).T
    scale = torch.reshape(scale, [21,1])
    # scale_expand = torch.tile(scale, [1,3])
    scale_expand = scale.repeat(1,3)
    uvd_scaled = uvd / scale_expand
    uvd_scaled = uvd_scaled * depth

    return uvd_scaled

def xyz_transform(xyz_ref, ext): #ref to cam
    device = xyz_ref.device
    # xyz_ref = torch.concat((xyz_ref.T, torch.ones([1,21]).to(device)), axis=0)
    xyz_ref = torch.cat((xyz_ref.T, torch.ones([1,21]).to(device)), axis=0)
    # xyz_cam = torch.linalg.matmul(ext, xyz_ref) 
    xyz_cam = torch.matmul(ext, xyz_ref) 
    return xyz_cam.T[:, :-1]

def uvdtouvd(uvd_a, K_a, K_b, extb_a):
    device = uvd_a.device
    K_a = torch.FloatTensor(K_a).to(device)
    K_b = torch.FloatTensor(K_b).to(device)
    extb_a = torch.FloatTensor(extb_a).to(device)
    uvd_b = xyztouvd(xyz_transform(uvdtoxyz(uvd_a, K_a), extb_a),K_b)
    return uvd_b



if __name__ == "__main__":
    import json   
    with open('./test_tw/onlyHandWorldCoordinate_uvd.json', 'r') as f:
        hand_uvd = json.loads(f.read())
    with open('./test_tw/onlyHandWorldCoordinate_xyz.json', 'r') as f:
        hand_xyz = json.loads(f.read())
    with open('/root/OXR_projects/optimizer/extrinsic.json', 'r') as f:
        ext = json.load(f) #0_ref, 1_ref, 2_ref
    Ks = np.load('./intrinsic.npy')
    joint0_uvd = np.array(hand_uvd['0_0'])
    a = uvdtouvd(joint0_uvd, Ks[0], Ks[1], ext['ref0'][1])
    print(a.shape)