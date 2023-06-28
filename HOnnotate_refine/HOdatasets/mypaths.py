import os

# OXR_MULTI_CAMERA_DIR = '/home/uvrlab/projects/HOnnotate_OXR/dataset/'
#
# YCB_MODELS_DIR = '/home/uvrlab/projects/HOnnotate_OXR/HOnnotate_refine/YCBobject/models/'
YCB_OBJECT_CORNERS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../objCorners')
MANO_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../optimization/mano/models/MANO_RIGHT.pkl')