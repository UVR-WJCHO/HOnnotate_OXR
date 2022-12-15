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