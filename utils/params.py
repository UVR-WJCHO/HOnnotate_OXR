import numpy as np
import torch

glob_rot_range = 6.28
rot_min_list = [
    -glob_rot_range, -glob_rot_range, -glob_rot_range,                  #0
    -0.25, -0.5, -1.0,                                                  #1
    0.0, 0.0, -1.0,                                                     #2
    0.0, 0.0, -0.3,                                                     #3
    -0.25, -0.5, -1.0,                                                  #4
    0.0, 0.0, -1.0,                                                     #5
    0.0, 0.0, -0.2,                                                     #6
    -0.25, -0.5, -1.25,                                                 #7
    -1.0, 0.0, -0.75,                                                   #8
    0.0, 0.0, -0.75,                                                    #9
    -0.25, -0.5, -1.0,                                                  #10
    0.0, 0.0, -1.0,                                                     #11
    0.0, 0.0, -0.5,                                                     #12
    -0.5, -1.0, -1.0,                                                   #13
    0.0, -1.0, 0.0,                                                     #14
    0.0, -1.5, -0.5                                                     #15
]

rot_max_list = [   
    glob_rot_range, glob_rot_range, glob_rot_range,                     #0
    0.25, 0.5, 1.5,                                                     #1
    0.0, 0.0, 1.5,                                                      #2
    0.0, 0.0, 1.5,                                                      #3
    0.25, 0.5, 1.5,                                                     #4
    0.0, 0.0, 1.5,                                                      #5
    0.0, 0.0, 1.5,                                                      #6
    0.25, 0.5, 1.5,                                                     #7
    1.0, 0.0, 1.5,                                                      #8
    0.0, 0.0, 1.5,                                                      #9
    0.25, 0.5, 1.5,                                                     #10
    0.0, 0.0, 1.5,                                                      #11
    0.0, 0.0, 1.5,                                                      #12
    -0.5, 0.5, 1.5,                                                     #13
    0.0, 0.5, 0.0,                                                      #14
    0.0, 1.0, -0.5                                                      #15
]

pose_mean_list = [
    0.012990133975529916, -0.2061533137671232, 0.1217358369522687, 
    0.0, 0.0, -0.26776195873424913, 
    0.0, 0.0, 0.1463492324860077, -0.019004827777825577, 
    0.045829039124687755, 0.008727056210913353, 0.0, 
    0.0, -0.1284752811149675, 
    0.0, 0.0, 0.23133631452056908, 
    -0.026223380872148847, 0.13406619518188134, -0.24682822100890117, 
    0.08579138535571655, 0.0, -0.08518110410666833, 
    0.0, 0.0, 0.5096552875298644, 
    -0.024652840864526005, -0.030070189038857012, -0.026405485406310773, 
    0.0, 0.0, -0.10829962287750862, 
    0.0, 0.0, 0.2650733518398053, 
    -0.5, -0.27061772045666727, 0.26407928733983543, 
    0.0, 0.07141815208289819, 0.0, 
    0.0, -0.23984496193594806, -0.5
]

punish_cent_joint0 = 10000.0
punish_cent_joint1 = 10000.0
punish_cent_joint2 = 10000.0

punish_side_joint0 = 0.0
punish_side_joint1 = 10000.0
punish_side_joint2 = 10000.0

punish_vert_joint0 = 0.0
punish_vert_joint1 = 0.0
punish_vert_joint2 = 0.0

pose_reg_list = [
    # Finger: Index
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0,     
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,            
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,   
    # Finger: Middle        
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0,     
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,                     
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,   
    # Finger: Ring                
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0,     
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,                     
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,    
    # Finger: Pinky            
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0, 
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,                     
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,#38   
    # Finger: Thumb    
    punish_cent_joint0, punish_vert_joint0, punish_cent_joint0, 
    punish_cent_joint1, punish_vert_joint1, punish_side_joint1,
    punish_cent_joint2, punish_vert_joint2, punish_side_joint2                       
]

initial_rot = torch.tensor([[-0.9538, -1.6024, 1.4709]])
initial_pose = torch.tensor([[0.0211, -0.3187, 0.0651, 0.0000, 0.0000, -0.6944, 0.0000, 0.0000,
                 0.2245, -0.0072, -0.2008, 0.2014, 0.0000, 0.0000, -0.7271, 0.0000,
                 0.0000, 0.2305, -0.0091, -0.1125, -0.1157, 0.0883, 0.0000, -0.5734,
                 0.0000, 0.0000, 0.1881, -0.0093, -0.3019, 0.0766, 0.0000, 0.0000,
                 -0.6934, 0.0000, 0.0000, 0.3669, -0.5000, -0.2614, 0.2612, 0.0000,
                 -0.5810, 0.0000, 0.0000, 0.8677, -0.5000]])

colors = [
    [0, 255, 0],		#0
    [0, 223, 0],		#1
    [0, 191, 0],		#2
    [0, 159, 0],		#3

    [159, 255, 0],		#4
    [159, 223, 0],		#5
    [159, 191, 0],		#6
    [159, 159, 0],		#7

    [255, 0, 0], 		#8
    [223, 0, 0],		#9
    [191, 0, 0],	 	#10
    [159, 0, 0], 		#11

    [255, 0, 255],	 	#12
    [255, 0, 223],		#13
    [255, 0, 191],	 	#14
    [255, 0, 159],		#15

    [0, 0, 255], 		#16
    [0, 0, 223], 		#17
    [0, 0, 191], 		#18
    [0, 0, 159],		#19
]

limbSeq_hand = [
    [0, 1],		#Thumb1
    [1, 2],		#Thumb2
    [2, 3],		#Thumb3
    [3, 4],		#Thumb4

    [0, 5],		#index1
    [5, 6],		#index2
    [6, 7],		#index3
    [7, 8],		#index4

    [0, 9],		#middle1
    [9, 10],	#middle2
    [10 ,11],	#middle3
    [11, 12],	#middle4

    [0, 13],	#ring1
    [13, 14],	#ring2
    [14, 15],	#ring3
    [15, 16],	#ring4

    [0, 17],	#pinky1
    [17, 18],	#pinky2
    [18, 19],	#pinky3
    [19, 20]	#pinky4
]