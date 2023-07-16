import os
import numpy as np

MANO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'manopth/mano/models/')

ORIGIN_WIDTH = 1920
ORIGIN_HEIGHT = 1080

IMG_WIDTH = 640
IMG_HEIGHT = 480

fin4_ver_idx1 = 23
fin4_ver_idx2 = fin4_ver_idx1+3
fin4_ver_fix_idx = 24

mano_key_bone_len = 10.0