import os

## Config
CFG_DATA_DIR = os.path.join(os.getcwd(), 'dataset')

CFG_LR_INIT = 0.02
CFG_LR_INIT_OBJ = 0.02

CFG_NUM_ITER = 100
CFG_DEVICE = 'cuda'
CFG_BATCH_SIZE = 1
CFG_MANO_PATH = os.path.join(os.getcwd(), 'modules', 'mano', 'models')

CFG_LOSS_DICT = ['kpts2d', 'reg']#, 'depth', 'depth_obj'] #['kpts2d', 'reg', 'seg']
CFG_SAVE_PATH = os.path.join(os.getcwd(), 'output')
CFG_CAMID_SET = ['mas', 'sub1', 'sub2', 'sub3']

CFG_IMG_WIDTH = 1920
CFG_IMG_HEIGHT = 1080
CFG_CROP_IMG_WIDTH = 640
CFG_CROP_IMG_HEIGHT = 480

CFG_WITH_OBJ = False