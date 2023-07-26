import cv2
import numpy as np
import pickle
import json


pkl_name = '/home/uvr-1080ti/projects/HOnnotate/sequence/xrstudio/OurHand_init/hand_result/bowl_18_00/handDetection_uvd.json'
img_name = '/home/uvr-1080ti/projects/HOnnotate/sequence/xrstudio/rgb/original/mas_4.png'


img_idx = 4
with open(pkl_name, 'rb') as f:

    result = json.load(f)
    mas_mas_2d = np.asarray(result['mas_mas'])[:, :, :2]

    sample = mas_mas_2d[4]

    img = cv2.imread(img_name)

    for i in range(21):
        cv2.circle(img, (int(sample[i][0]), int(sample[i][1])), 3, (0, 0, 255), -1)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
    
def orthographic_proj_withz(X, trans, scale, offset_z=0.):
    """
    X: B x N x 3
    trans: B x 2: [tx, ty]
    scale: B x 1: [sc]
    Orth preserving the z.
    """
    scale = scale.contiguous().view(-1, 1, 1)
    trans = trans.contiguous().view(scale.size(0), 1, -1)
    proj = scale * X

    proj_xy = proj[:, :, :2] + trans[:,:,:2]
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)