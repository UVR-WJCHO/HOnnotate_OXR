import os
import pyk4a
from pyk4a import PyK4A, Config
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from socket import *
import math
import winsound as sd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="230802")
args = parser.parse_args()

outfolder = args.dir
os.makedirs(outfolder, exist_ok=True)

camSet = ['mas', 'sub1', 'sub2', 'sub3']

current_fps = 2

master_config = Config(synchronized_images_only=True, wired_sync_mode=1,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=-80,)
sub1_config = Config(synchronized_images_only=True, wired_sync_mode=2,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=80,)
sub2_config = Config(synchronized_images_only=True, wired_sync_mode=2,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=80+160,)
sub3_config = Config(synchronized_images_only=True, wired_sync_mode=2,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=80+160+160,)

# Load camera with the default config
k4a_master = PyK4A(device_id=0, config=master_config)
k4a_sub1 = PyK4A(device_id=1, config=sub1_config)
k4a_sub2 = PyK4A(device_id=2, config=sub2_config)
k4a_sub3 = PyK4A(device_id=3, config=sub3_config)

k4a_list = [k4a_master, k4a_sub1, k4a_sub2, k4a_sub3]

k4a_sub3.start()
k4a_sub2.start()
k4a_sub1.start()
k4a_master.start()
print(k4a_master.serial)
print(k4a_sub1.serial)
print(k4a_sub2.serial)
print(k4a_sub3.serial)
print(k4a_master.calibration.get_camera_matrix(camera=1))
print(k4a_sub1.calibration.get_camera_matrix(camera=1))
print(k4a_sub2.calibration.get_camera_matrix(camera=1))
print(k4a_sub3.calibration.get_camera_matrix(camera=1))

k4a_master.save_calibration_json(path=str(args.dir) + '/' + 'mas_camInfo.json')
k4a_sub1.save_calibration_json(path=str(args.dir) + '/' + 'sub1_camInfo.json')
k4a_sub2.save_calibration_json(path=str(args.dir) + '/' + 'sub2_camInfo.json')
k4a_sub3.save_calibration_json(path=str(args.dir) + '/' + 'sub3_camInfo.json')

print("distortion_coefficients")
print(k4a_master.calibration.get_distortion_coefficients(camera=1))
print(k4a_sub1.calibration.get_distortion_coefficients(camera=1))
print(k4a_sub2.calibration.get_distortion_coefficients(camera=1))
print(k4a_sub3.calibration.get_distortion_coefficients(camera=1))

f_name = str(args.dir) + '/' + 'cameraInfo.txt'
with open(f_name,"w") as f:
    for camname in camSet:
        f.write(camname + '\n')
    f.write('\n')

    f.write('device_serial\n')
    for i in range(len(camSet)):
        f.write(str(k4a_list[i].serial) + '\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(len(camSet)):
        f.writelines(str(k4a_list[i].calibration.get_camera_matrix(camera=1)))
        f.write('\n')
