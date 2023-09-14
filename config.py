import os
from enum import IntEnum


## Debug Flags ##

CFG_WITH_OBJ = False
CFG_EARLYSTOPPING = False

CFG_LOSS_DICT = ['kpts2d', 'reg', 'depth_rel']#, 'seg', 'depth']#, ] #, 'contact', 'temporal']

CFG_LR_INIT = 0.1
CFG_LR_INIT_OBJ = 0.1

CFG_DEPTH_RANGE = {'mas':[500, 1000], 'sub1':[200, 750], 'sub2':[0, 1100], 'sub3':[200, 900]}
CFG_CAM_WEIGHT = [1.0, 1.0, 1.0, 1.0]

CFG_CAM_PER_FINGER_VIS = {'mas':[1.0, 1.0,1.0,1.0,1.0],
                         'sub1':[1.0,0.3,0.3,1.0,1.0],
                         'sub2':[1.0, 1.0,1.0,1.0,1.0],
                         'sub3':[0.7,1.0,1.0,1.0,1.0]}


CFG_vertspermarker = {
    "mug" : [1282, 1329, 965, 756],
    "cardboard_box" : [39,1313,716,1294],
}


## Config
CFG_DATA_DIR = os.path.join(os.getcwd(), 'dataset')
CFG_CAMID_SET = ['mas', 'sub1', 'sub2', 'sub3']

CFG_NUM_ITER = 150
CFG_DEVICE = 'cuda'
CFG_BATCH_SIZE = 1
CFG_MANO_PATH = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
CFG_MANO_SIDE = 'right'

CFG_SAVE_PATH = os.path.join(os.getcwd(), 'output')

CFG_IMG_WIDTH = 1920
CFG_IMG_HEIGHT = 1080
CFG_CROP_IMG_WIDTH = 640
CFG_CROP_IMG_HEIGHT = 480

CFG_LOSS_THRESHOLD = 3500
CFG_PATIENCE = 30
CFG_PATIENCE_v2 = 50

CFG_CONTACT_START_THRESHOLD = 15000 # use contact loss when kpts_loss < 15000
CFG_CONTACT_DIST = 8
CFG_CONTACT_LOSS_WEIGHT = 4

CFG_PALM_IDX = [0, 5, 9, 13]

class GRASPType(IntEnum):
    '''
    Enum for different datatypes
    '''

    Large_Diameter = 1
    Small_Diameter = 2
    Medium_Wrap = 3
    Adducted_Thumb = 4
    Light_Tool = 5
    Prismatic_4_Finger = 6
    Prismatic_3_Finger = 7
    Prismatic_2_Finger = 8
    Palmar_Pinch = 9
    Power_Disk = 10
    Power_Sphere = 11
    Precision_Disk = 12
    Precision_Sphere = 13
    Tripod = 14
    Fixed_Hook = 15
    Lateral = 16
    Index_Finger_Extension = 17
    Extension_Type = 18
    Distal = 19
    Writing_Tripod = 20
    Tripod_Variation = 21
    Parallel_Extension = 22
    Adduction_Grip = 23
    Tip_Pinch = 24
    Lateral_Tripod = 25
    Sphere_4_Finger = 26
    Quadpod = 27
    Sphere_3_Finger = 28
    Stick = 29
    Palmar = 30
    Ring = 31
    Ventral = 32
    Inferior_Pincer = 33


class OBJType(IntEnum):
    cracker_box = 1
    potted_meat_can = 2
    banana = 3
    apple = 4
    wine_glass = 5
    bowl = 6
    mug = 7
    plate = 8
    spoon = 9
    knife = 10
    small_marker = 11
    spatula = 12
    flat_screwdriver = 13
    hammer = 14
    baseball = 15
    golf_ball = 16
    credit_card = 17
    dice = 18
    dist_lid = 19
    smartphone = 20
    mouse = 21
    tape = 22
    master_chef_can = 23
    Scrub_cleanser_bottle = 24
    large_marker = 25
    stapler = 26
    note = 27
    scissors = 28
    foldable_phone = 29
    cardboard_box = 30
