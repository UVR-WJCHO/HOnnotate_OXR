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
parser.add_argument('--dir', type=str, default="230822_S01_obj_01")
# --split {trial_0 start frame} {trial_1 start frame} ... {trial_n start frame} {trial_n end frame}
parser.add_argument('--split', type=int, nargs='+', required=True)

# --grasp 5 17 20 (recorded grasp class)
parser.add_argument('--grasp', type=int, nargs='+', required=True)
# --trialnum 7 7 6 (each classes trial num)
parser.add_argument('--trialnum', type=int, nargs='+', required=True)

args = parser.parse_args()

def main():
    input_folder = args.dir
    #num_trials = args.trials
    split_list = args.split

    grasp_class_list = args.grasp
    trialnum_list = args.trialnum


    if not os.path.exists(os.path.join(input_folder, 'rgb')) or not os.path.exists(os.path.join(input_folder, 'depth')):
        raise Exception("no rgb, depth folder in input folder %s, check the --dir" % (input_folder))

    input_list = os.listdir(os.path.join(input_folder, 'rgb/mas'))
    input_list = natsorted(input_list)

    num_trials = len(split_list) - 1
    print("target split frames : ", split_list)

    grasp_idx = 0
    trial_idx = 0
    
    for i in range(num_trials):
    
        start_idx = split_list[i]
        end_idx   = split_list[i+1]

        # create grasp folder
        sub_grasp_name = input_folder + '_grasp_' + str(grasp_class_list[grasp_idx])
        sub_trial_name = 'trial_' + str(trial_idx)
        
        os.makedirs(os.path.join(input_folder, sub_grasp_name, sub_trial_name), exist_ok=True)
        os.makedirs(os.path.join(input_folder, sub_grasp_name, sub_trial_name, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(input_folder, sub_grasp_name, sub_trial_name, 'depth'), exist_ok=True)
        
        output_rgb_path = os.path.join(input_folder, sub_grasp_name, sub_trial_name, 'rgb')
        output_depth_path = os.path.join(input_folder, sub_grasp_name, sub_trial_name, 'depth')
        
        num = 0
        
        for i in range(start_idx,end_idx) :
            
            fail = 0
            
            for cam in ['mas','sub1','sub2','sub3'] :
                
                os.makedirs(os.path.join(input_folder, sub_grasp_name, sub_trial_name, 'rgb', cam), exist_ok=True)
                os.makedirs(os.path.join(input_folder, sub_grasp_name, sub_trial_name, 'depth', cam), exist_ok=True)
        
                
                src_rgb  = os.path.join(input_folder,'rgb',cam,cam+'_'+str(i)+'.jpg')
                dest_rgb = os.path.join(output_rgb_path,cam,cam+'_'+str(num)+'.jpg')
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
                    
            if fail == 0:
                num += 1

        trial_idx += 1
        if trial_idx == (trialnum_list[grasp_idx] - 1):
            trial_idx = 0
            grasp_idx += 1

       


if __name__ == '__main__':
    main()
