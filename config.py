import os
from enum import IntEnum


## Manual Flags ##

CFG_VALID_CAM = ['mas', 'sub1', 'sub2', 'sub3']
CFG_CAM_WEIGHT = [2.0, 1.0, 1.0, 1.0]

CFG_LR_INIT = 0.05
CFG_LR_INIT_OBJ = 1.0


CFG_NUM_ITER = 40

CFG_WITH_OBJ = True
CFG_EARLYSTOPPING = False

# set True if 2D tip annotation data exists(from euclidsoft)
CFG_exist_tip_db = False
CFG_LOSS_DICT = ['depth_obj', 'pose_obj'] # 'temporal','seg_obj',
# CFG_LOSS_DICT = ['reg', 'kpts2d', 'temporal', 'seg','depth', 'depth_obj', 'seg_obj', 'penetration', 'contact', 'pose_obj']#, 'kpts_tip']





if not CFG_exist_tip_db:
    assert 'kpts_tip' not in CFG_LOSS_DICT, 'need CFG_exist_tip_db=True'


CFG_LOSS_WEIGHT = {'kpts2d': 1.0, 'depth': 1.0, 'seg': 1.0, 'reg': 1.0, 'contact': 1.0, 'penetration': 1.0,
                   'depth_rel': 1.0, 'temporal': 1.0, 'kpts_tip':1.0, 'depth_obj': 1.0, 'seg_obj': 1.0, 'pose_obj':0.0}

CFG_temporal_loss_weight = 0.5e4

# given original images, tipGT generated for every 30 frames. We sample it to 1/3
CFG_tipGT_interval = 10

# default : False. only for sample
CFG_VIS_CONTACT = False
CFG_SAVE_MESH = False


CFG_DEPTH_RANGE = {'mas':[500, 1000], 'sub1':[200, 750], 'sub2':[0, 1100], 'sub3':[200, 900]}


# CFG_CAM_PER_FINGER_VIS = {'mas':[1.0, 1.0,1.0,1.0,1.0],
#                          'sub1':[1.5,0.5,0.5,0.5,0.5],
#                          'sub2':[1.0, 1.0,1.0,1.0,1.0],
#                          'sub3':[0.5,1.5,1.5,1.0,1.0]}
## apply on dataloader pkl creation
CFG_NON_VISIBLE_WEIGHT = 0.3

######## 230829-230908
# if included, .obj has mm scale
CFG_OBJECT_SCALE = ["05_wine_glass", "11_small_marker", "13_flat_screwdriver"]
CFG_OBJECT_SCALE_SPECIFIC = {"16_golf_ball":0.43, "05_wine_glass":0.95}

CFG_OBJECT_NO4th_POINT = ["24_Scrub_cleanser_bottle", "29_foldable_phone"]
CFG_OBJECT_4th_POINT = ["09_spoon"]



CFG_vertspermarker = {

    '230829~230908':
        {
    "01_cracker_box" : [1865,1859,4434,1070],
    "02_potted_meat_can" : [1157,1127,1102],
    "03_banana" : [158,204,1018],
    "04_apple" : [81,273,283],
    "05_wine_glass" : [1279 ,1982,354],
    "06_bowl" : [1010,536,548],
    "07_mug" : [862,1355,1322,],
    "08_plate" : [165,239,1000],
    "09_spoon" : [541, 535, 548],
    "10_knife" : [1780,1901,1771],
    "11_small_marker" : [1636, 2030, 2003],
    "12_spatula" : [33,103,115],
    "13_flat_screwdriver" : [162, 2824, 2816],
    "14_hammer" : [2195,2185,170,601],
    "15_baseball" : [1000,1304,525],
    "16_golf_ball" : [296,238,221],
    "17_credit_card" : [486,502,89,518],
    "18_dice" : [386,1102,1065],
    "19_disk_lid" : [1988,3366,3367],
    "20_smartphone" : [1352,828,1127],
    "21_mouse" : [46,582,1536],
    "22_tape" : [1544,1220,1671],
    "23_master_chef_can" : [4586,4597,4695],
    "24_Scrub_cleanser_bottle" : [2795,2948,2966],
    "25_large_marker" : [1152,1396,3007],
    "26_stapler" : [567,807,499],
    "27_note" : [334,3,733],
    "28_scissors" : [3932,3948,3407],
    "29_foldable_phone" : [[2002, 2005, 1728], [2718,2721,3087]],
    "30_cardboard_box" : [418,1314,413,1297]},

'230909~230913':
    {
    "01_cracker_box" : [1010,875,4434,2096],
    "02_potted_meat_can" : [1157,1127,410,1099],
    "03_banana" : [93,190,1018,74],
    "04_apple" : [679,825,659,832],
    "05_wine_glass" : [1187,1982,1942,1829,1247],
    "06_bowl" : [954,585,594,1116],
    "07_mug" : [862,1355,1325,918],
    "08_plate" : [1000,1080,1040,1282],
    "09_spoon" : [541,534,549,269],
    "10_knife" : [1759,1873,1921,1774],
    "11_small_marker" : [1636, 2030, 2003],
    "12_spatula" : [35,106,114,1106],
    "13_flat_screwdriver" : [162, 2824, 2816],
    "14_hammer" : [2195,2185,170,601,1266],
    "15_baseball" : [1251,912,1091,1643],
    "16_golf_ball" : [282,1404,51],
    "17_credit_card" : [486,502,89,518],
    "18_dice" : [386,1102,1065],
    "19_disk_lid" : [1863,3367,3568,2570,1680],
    "20_smartphone" : [1319,772,1140,2999],
    "21_mouse" : [46,582,1536, 359],
    "22_tape" : [1699,1172,1742, 1849],
    "23_master_chef_can" : [4586,4597,4695],#, 70, 30],
    "24_Scrub_cleanser_bottle" : [2872, 2948, 2966, 2805, 2873],
    "25_large_marker" : [1152,1124,3338],
    "26_stapler" : [567,807,499,1963, 516],
    "27_note" : [334, 755, 733, 1968, 1993],
    "28_scissors" : [3932,3948,3407, 307],
    "29_foldable_phone" : [[2002, 2005, 1728, 1342], [2718,2721,3087,3075]],
    "30_cardboard_box" : [418, 1314, 413, 1297, 91]},

'230914':
    {"01_cracker_box": [1010, 875, 4434, 2096],
     "02_potted_meat_can": [1157, 1127, 410, 1099],
     "03_banana": [93, 190, 1018, 74],
     "04_apple": [672, 821, 666, 918],
     "05_wine_glass": [1279, 2012, 2030, 1829, 1399],
     "06_bowl": [954, 585, 594, 1116],
     "07_mug": [862, 1355, 1325, 918],
     "08_plate": [1000, 1080, 1040, 1282],
     "09_spoon": [541, 534, 549, 269],
     "10_knife": [1759, 1873, 1921, 1774],
     "11_small_marker": [1636, 1974, 2074],
     "12_spatula": [37, 160, 172, 1106],
     "13_flat_screwdriver": [162, 2824, 2816],
     "14_hammer": [2195, 2185, 236, 601, 1266],
     "15_baseball": [835, 1147, 668, 1317],
     "16_golf_ball": [215, 122, 1412],
     "17_credit_card": [72, 56, 173, 57],
     "18_dice": [250, 748, 711],
     "19_disk_lid": [1863, 3367, 3468, 2665, 1785],
     "20_smartphone": [1431, 844, 1054, 2974],
     "21_mouse": [521, 1037, 1532, 248],
     "22_tape": [1473, 1175, 1741, 1379],
     "23_master_chef_can": [4586, 4580, 4646],
     "24_Scrub_cleanser_bottle": [5834, 2957, 2973, 5844],
     "25_large_marker": [1152, 1143, 1579],
     "26_stapler": [1937, 847, 601, 501],
     "27_note": [334, 3, 753, 1968, 1984],
     "28_scissors": [3931, 4101, 3992, 305],
     "29_foldable_phone": [[1513, 1557, 1742, 1665], [2721, 2718, 2458, 3074]],
     "30_cardboard_box": [406, 1284, 418, 1314, 97],
     },
# not checked
'230915~':
    { "01_cracker_box" : [1010,875,4434,2096],
    "02_potted_meat_can" : [1157,1127,410,1099],
    "03_banana" : [93,190,1018,74],
    "04_apple" : [681,820,925,830],
    "05_wine_glass" : [1279,2012,2030,1829,1399],#[1184,1354,354,1829,1245],
    "06_bowl" : [954,585,594,1116],
    "07_mug" : [862,1355,1325,918],
    "08_plate" : [1000,1080,1040,1282],
    "09_spoon" : [541,534,549,269],
    "10_knife" : [1759,1873,1921,1774],
    "11_small_marker" : [2162,2000,2102],
    "12_spatula" : [37,160,172,1106],
    "13_flat_screwdriver" : [162, 2824, 2816],
    "14_hammer" : [2195,2185,236,601,1266],
    "15_baseball" : [1317,835,668,12],
    "16_golf_ball" : [215,122,1412],
    "17_credit_card" : [72,56,173,57],
    "18_dice" : [204,1065,1102],
    "19_disk_lid" : [1863,3367,3468,2665,1785],
    "20_smartphone" : [1431,844,1054,2974],
    "21_mouse" : [521,1037,1532,248],
    "22_tape" : [1473,1175,1741,1379],
    "23_master_chef_can" : [4586,4580,4646],#,x,x],
    "24_Scrub_cleanser_bottle" : [5834,2957,2973,5844],#,2873],
    "25_large_marker" : [1152,1311,3361],
    "26_stapler" : [1937,847,601,501],
    "27_note" : [334,3,753,1968,1984],
    "28_scissors" : [3931,4101,3992,305],
    "29_foldable_phone" : [[1513,1557,1742,1665], [2721,2718,2458,3074]],
    "30_cardboard_box" : [406,1284,418,1314,97],
}

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
CFG_PATIENCE_obj = 4

CFG_CONTACT_START_THRESHOLD = 5000 # use contact loss when kpts_loss < 5000
CFG_CONTACT_DIST = 8
CFG_CONTACT_DIST_VIS = 10

CFG_PALM_IDX = [0, 5, 9, 13]
CFG_TIP_IDX = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20}

CFG_OBJECT_SCALE_FIXED = [1.,
1.,
1.,
1.,
0.8296698468,
1.,
1.,
1.,
1.,
1.,
0.1035083229,
1.,
0.6706711338,
1.,
1.,
0.43,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.,
1.]

CAM_calibration_error = {
230904:	1.47946351,
230905:	1.659291,
230906:	1.49168,
230907:	0.146457,
230907_2:	2.10217,
230908:	0.3123178,
230910:	1.465127,
230911:	1.4393,
230911_2:	2.0714158,
230912:	0.3015487,
230913:	0.205707,
230914:	0.453806,
230914_2:	1.587009,
230915:	0.34341,
230916:	0.7792153,
230917:	4.15488,
230918:	1.68417,
230919:	4.386784,
230920:	1.4454,
230921:	0.34172213,
230922:	0.304287,
230922_2:	0.32866,
230922_3:	0.40456,
230922_4:	0.47551,
230923:	0.938846,
230924:	1.73025,
230925:	1.631825,
230926:	1.791621,
230927:	0.349239,
230927_2:	0.33341,
231001:	0.566916,
231001_2:	0.81117,
231002:	0.30335,
231002_2:	2.10091,
231003:	0.853757,
231004:	0.31779,
231005:	0.56786,
231005_2:	0.291308,
231006:	0.58833,
231006_2:	0.202002,
231007:	1.786432,
231007_2:	4.065617,
231008:	1.0508705,
231008_2:	1.4544749,
231008_3:	0.937868,
231009:	1.653466,
231009_2:	1.05663,
231010:	0.6928807,
231010_2:	0.806954,
231011:	1.0557944,
231011_2:	0.85368602,
231011_3:	1.2349034,
231012:	0.99202,
231012_2:	0.1785122,
231013:	0.681108,
231013_2:	0.27652407,
231014:	0.100819,
231014_2:	0.088672,
231015:	0.150844,
231015_2:	0.2477184,
231016:	0.362411,
231016_2:	0.2529165,
231017:	0.12093,
231017_2:	0.200956438,
231018:	0.1965513,
231018_2:	0.2803081,
231019:	1.920469,
231019_2:	0.124112,
231019_3:	0.3123295,
231019_4:	0.207655,
231020:	1.55187,
231021:	0.4701261,
231021_2:	0.177197,
231021_3:	0.21288,
231023:	0.15478,
231023_2:	0.11686,
231024:	0.167597,
231024_2:	0.1046901,
231025:	0.337806028,
231026:	0.14738,
231026_2:	0.234149,
231027:	0.168592,
}



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
