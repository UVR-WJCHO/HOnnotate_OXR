import os

HO3D_MULTI_CAMERA_DIR = '/home/uvr-1080ti/projects/HOnnotate_OXR/HOnnotate_refine/sequence'
OXR_MULTI_CAMERA_DIR = '/home/uvr-1080ti/projects/HOnnotate_OXR/dataset/'

YCB_MODELS_DIR = '/home/uvr-1080ti/projects/HOnnotate_OXR/HOnnotate_refine/YCB_Models/models/'

YCB_OBJECT_CORNERS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../objCorners')

MANO_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../optimization/mano/models/MANO_RIGHT.pkl')