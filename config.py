import os
from enum import IntEnum


## Debug Flags ##


CFG_WITH_OBJ = True
CFG_EARLYSTOPPING = False


# set True if 2D tip annotation data exists(from euclidsoft)
CFG_exist_tip_db = False
# default : False. only for sample
CFG_VIS_CONTACT = False
CFG_SAVE_MESH = False

CFG_LOSS_DICT = ['reg', 'kpts2d', 'temporal', 'seg','depth', 'depth_obj', 'seg_obj', 'penetration', 'contact']#, , 'pose_obj']#, 'kpts_tip']#,,,] # 'depth_rel',  'contact',

if not CFG_exist_tip_db:
    assert 'kpts_tip' not in CFG_LOSS_DICT, 'need CFG_exist_tip_db=True'

CFG_LOSS_WEIGHT = {'kpts2d': 1.0, 'depth': 1.0, 'seg': 1.0, 'reg': 1.0, 'contact': 1.0, 'penetration': 1.0,
                   'depth_rel': 1.0, 'temporal': 1.0, 'kpts_tip':1.0, 'depth_obj': 1.0, 'seg_obj': 1.0, 'pose_obj':1.0}

CFG_temporal_loss_weight = 0.5e5

# given original images, tipGT generated for every 30 frames. We sample it to 1/3
CFG_tipGT_interval = 10

CFG_LR_INIT = 0.05
CFG_LR_INIT_OBJ = 0.01

CFG_NUM_ITER = 75

CFG_DEPTH_RANGE = {'mas':[500, 1000], 'sub1':[200, 750], 'sub2':[0, 1100], 'sub3':[200, 900]}
CFG_CAM_WEIGHT = [1.0, 1.0, 1.0, 1.0]


# CFG_CAM_PER_FINGER_VIS = {'mas':[1.0, 1.0,1.0,1.0,1.0],
#                          'sub1':[1.5,0.5,0.5,0.5,0.5],
#                          'sub2':[1.0, 1.0,1.0,1.0,1.0],
#                          'sub3':[0.5,1.5,1.5,1.0,1.0]}
## apply on dataloader pkl creation
CFG_NON_VISIBLE_WEIGHT = 0.5



######## 230829-230908
# if included, .obj has mm scale
CFG_OBJECT_SCALE = {"11_small_marker":1.0, "13_flat_screwdriver":6.5, "16_golf_ball":1.0}

CFG_vertspermarker = {
    "01_cracker_box" : [1865,1859,4434,1070],
    "02_potted_meat_can" : [1157,1127,410],
    "03_banana" : [158,204,1018],
    "04_apple" : [81,273,283],
    "05_wine_glass" : [264 ,434,354], #[1195,1950,1990],    494
    "06_bowl" : [1010,536,548],
    "07_mug" : [862,1355,1322,],# [1363,1571,1346,],
    "08_plate" : [3400,3480,1040],
    "09_spoon" : [541, 535, 548], #[604,620,610],
    "10_knife" : [1780,1901,1771],
    "11_small_marker" : [1636, 2030, 2003],#[1636,2131,2003],
    "12_spatula" : [33,103,115],
    "13_flat_screwdriver" : [162, 2824, 2816],#[271,2817,2813],
    "14_hammer" : [2195,2185,170,601],
    "15_baseball" : [1000,1304,525],
    "16_golf_ball" : [296,238,221], # [1120,1016,952],
    "17_credit_card" : [486,502,89,518],
    "18_dice" : [386,1102,1065],
    "19_disk_lid" : [1988,3367,3366],
    "20_smartphone" : [1352,828,1127],
    "21_mouse" : [46,582,1536],
    "22_tape" : [1665,1299,1693],
    "23_master_chef_can" : [4586,4597,4695],
    "24_Scrub_cleanser_bottle" : [2795,2948,2966],
    "25_large_maker" : [1152,1176,3018],
    "26_stapler" : [662,12276,11712],
    "27_note" : [334,0,7449],
    "28_scissors" : [1112,6900,1609],
    "29_foldable_phone" : [2002,2005,1728],
    "30_cardboard_box" : [418,1314,413,1297]
}

####################



CFG_valid_index = [[0, 1,2, 5,6, 9,10, 13,14, 17,18],
                   [0, 1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19]]

## Config
CFG_ROOT_DIR = os.path.join(os.getcwd())
CFG_DATA_DIR = os.path.join(os.getcwd(), 'dataset')
CFG_CAMID_SET = ['mas', 'sub1', 'sub2', 'sub3']


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
CFG_PATIENCE_obj = 3

CFG_CONTACT_START_THRESHOLD = 5000 # use contact loss when kpts_loss < 5000
CFG_CONTACT_DIST = 8
CFG_CONTACT_DIST_VIS = 10

CFG_PALM_IDX = [0, 5, 9, 13]
CFG_TIP_IDX = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20}



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
    disk_lid = 19
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
