import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))
import glob
import argparse
import json
import numpy as np
import math
import cv2
from natsort import natsorted

import shutil


### split points are start of new folder ###

### python sampleSplit.py --dir 230822 --trials 7 --split 15 85 150 ...

# split에 start,end로 구성되도록 작성함.

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="230802")
#parser.add_argument('--trials', type=int, default=7)
parser.add_argument('--split', type=int, nargs='+', required=True)
args = parser.parse_args()

def main():
    input_folder = args.dir
    #num_trials = args.trials
    split_list = args.split

    if not os.path.exists(os.path.join(input_folder, 'rgb')) or not os.path.exists(os.path.join(input_folder, 'depth')):
        raise Exception("no rgb, depth folder in input folder %s, check the --dir" % (input_folder))

    input_list = os.listdir(os.path.join(input_folder, 'rgb/mas'))
    input_list = natsorted(input_list)

    
    num_trials = len(split_list)
    
    split_list.append( int(input_list[-1].split('_')[1].split('.')[0]) + 1 )
    
    print("target split frames : ", split_list)
    
    for i in range(num_trials):
    
        start_idx = split_list[i]
        end_idx   = split_list[i+1]
        
        sub_name = 'trial_' + str(i)
        
        os.makedirs(os.path.join(input_folder, sub_name), exist_ok=True)
        os.makedirs(os.path.join(input_folder, sub_name, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(input_folder, sub_name, 'depth'), exist_ok=True)
        
        output_rgb_path = os.path.join(input_folder, sub_name, 'rgb')
        output_depth_path = os.path.join(input_folder, sub_name, 'depth')
        
        num = 0
        
        for i in range(start_idx,end_idx) :
            
            fail = 0
            
            for cam in ['mas','sub1','sub2','sub3'] :
                
                os.makedirs(os.path.join(input_folder, sub_name, 'rgb', cam), exist_ok=True)
                os.makedirs(os.path.join(input_folder, sub_name, 'depth', cam), exist_ok=True)
        
                
                src_rgb  = os.path.join(input_folder,'rgb',cam,cam+'_'+str(i)+'.png')
                dest_rgb = os.path.join(output_rgb_path,cam,cam+'_'+str(num)+'.png')
                src_depth  = os.path.join(input_folder,'depth',cam,cam+'_'+str(i)+'.png')
                dest_depth = os.path.join(output_depth_path,cam,cam+'_'+str(num)+'.png')
                
                try :
                    shutil.copy(src_rgb, dest_rgb)
                    shutil.copy(src_depth, dest_depth)
                    
                except :
                    print('missing',src_rgb)
                    print('missing',src_depth)
                    fail = 1
                    continue
                    
            if fail == 0 :
                num += 1
            
                            
        # frame_curr = int(split_list[i])
        # frame_next = int(split_list[i+1])
        # while True:
        #     if parse_idx >= frame_curr:
        #         src_rgb = input_rgb[frame_curr]
        #         src_depth = input_depth[frame_curr]
        #
        #         src_rgb_name = os.path.split(src_rgb)[-1]
        #         src_rgb_cam = src_rgb_name[:-4]
        #
        #         output_rgb_name = src_rgb_cam + '_'
        #         output_rgb = os.path.join(output_rgb_path, )
        #
        #     parse_idx += 1
        

       


if __name__ == '__main__':
    main()
