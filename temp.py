import os
import pickle
from natsort import natsorted
import numpy as np
import cv2
import matplotlib.pyplot as plt


rootDir = "E:/HOnnotate_OXR/depth_example/depth"
outDir = "E:/HOnnotate_OXR/depth_example/depth_vis"
cam_list = natsorted(os.listdir(rootDir))

cm = plt.get_cmap('nipy_spectral')

for camIdx, camName in enumerate(cam_list):
    camDir = os.path.join(rootDir, camName)
    img_list = natsorted(os.listdir(camDir))

    for imgIdx, imgName in enumerate(img_list):
        imgPath = os.path.join(rootDir, camName, imgName)
        depth_raw = np.asarray(cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)).astype(float)

        depth_raw[depth_raw>1000] = 0

        depth_max = np.max(depth_raw)
        depth_vis = depth_raw / depth_max
        colored_image = cm(depth_vis)
        depth_vis = (colored_image[:, :, :3] * 255).astype(np.uint8)
        # cv2.imshow("depth_vis", depth_vis)
        # cv2.waitKey(0)

        outPath = os.path.join(outDir, camName, imgName)
        cv2.imwrite(outPath, depth_vis)

