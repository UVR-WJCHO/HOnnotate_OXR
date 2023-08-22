## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import os
# First import the library
import pyk4a
from pyk4a import PyK4A, Config
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# import winsound as sd
import argparse

import time

import multiprocessing
from multiprocessing import Queue


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="230802")
parser.add_argument('--num', type=int, default=100)
args = parser.parse_args()

outfolder = args.dir
os.makedirs(outfolder, exist_ok=True)
os.makedirs(os.path.join(outfolder,'rgb'), exist_ok=True)
os.makedirs(os.path.join(outfolder,'depth'), exist_ok=True)

camList = ['mas', 'sub1', 'sub2', 'sub3']

for cam in camList:
    os.makedirs(os.path.join(outfolder, 'rgb', cam), exist_ok=True)
    os.makedirs(os.path.join(outfolder, 'depth', cam), exist_ok=True)

# Set camera config
current_fps=0   # 1 : 15fps # 2: 30fps
num_frames = args.num

master_config = Config(synchronized_images_only=True, wired_sync_mode=1,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=-80,)
sub1_config = Config(synchronized_images_only=True, wired_sync_mode=2,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=80,)
sub2_config = Config(synchronized_images_only=True, wired_sync_mode=2,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=80+160,)
sub3_config = Config(synchronized_images_only=True, wired_sync_mode=2,camera_fps=current_fps, color_resolution=pyk4a.ColorResolution.RES_1080P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,depth_delay_off_color_usec=80+160+160,)

# Load camera with the default config
k4a_master = PyK4A(device_id=0, config=master_config)
k4a_sub1 = PyK4A(device_id=1, config=sub1_config)
k4a_sub2 = PyK4A(device_id=2, config=sub2_config)
k4a_sub3 = PyK4A(device_id=3, config=sub3_config)


def run_capture(queue):

    # must start sub first
    k4a_sub3.start()
    k4a_sub2.start()
    k4a_sub1.start()
    k4a_master.start()

    try:
        for i in range(num_frames):
            start_time = time.time()
            capture = k4a_master.get_capture()
            capture_1 = k4a_sub1.get_capture()
            capture_2 = k4a_sub2.get_capture()
            capture_3 = k4a_sub3.get_capture()

            color = capture.color
            depth = capture.transformed_depth
            color_1 = capture_1.color
            depth_1 = capture_1.transformed_depth
            color_2 = capture_2.color
            depth_2 = capture_2.transformed_depth
            color_3 = capture_3.color
            depth_3 = capture_3.transformed_depth

            queue.put(
                (color, depth, color_1, depth_1, color_2, depth_2, color_3, depth_3)
            )

            diff = time.time() - start_time
            print('%d.png' % i + ' captured' + ' fps: ' + str(1/diff))
    except KeyboardInterrupt:
        k4a_sub3.stop()
        k4a_sub2.stop()
        k4a_sub1.stop()
        k4a_master.stop()
    finally:
        k4a_sub3.stop()
        k4a_sub2.stop()
        k4a_sub1.stop()
        k4a_master.stop()



if __name__=="__main__":

    queue = Queue()

    process_capture = multiprocessing.Process(target=run_capture, args=(queue,))
    process_capture.start()

    rgb_path = os.path.join(outfolder, 'rgb')
    depth_path = os.path.join(outfolder, 'depth')

    while True:
        if queue.empty():
            time.sleep(0.5)
            continue
        else:
            break
    try:
        for i in range(num_frames):
            color, depth, color_1, depth_1, color_2, depth_2, color_3, depth_3 = queue.get()
            cv2.imwrite(rgb_path + '/mas/mas_' + str(i) + '.jpg', color)
            cv2.imwrite(depth_path + '/mas/mas_' + str(i) + '.png', depth)

            cv2.imwrite(rgb_path + '/sub1/sub1_' + str(i) + '.jpg', color_1)
            cv2.imwrite(depth_path + '/sub1/sub1_' + str(i) + '.png', depth_1)

            cv2.imwrite(rgb_path + '/sub2/sub2_' + str(i) + '.jpg', color_2)
            cv2.imwrite(depth_path + '/sub2/sub2_' + str(i) + '.png', depth_2)

            cv2.imwrite(rgb_path + '/sub3/sub3_' + str(i) + '.jpg', color_3)
            cv2.imwrite(depth_path + '/sub3/sub3_' + str(i) + '.png', depth_3)
            print('%d.png' % i + ' saved')
    except KeyboardInterrupt:
        process_capture.terminate()
    finally:
        process_capture.terminate()
