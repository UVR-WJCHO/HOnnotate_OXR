import os
import sys
sys.path.insert(0,os.path.join(os.getcwd()))

import torch
import torch.nn as nn
import numpy as np
import cv2

from modules.dataloader import DataLoader, ObjectLoader
from modules.renderer import Renderer
from modules.meshModels import HandModel, ObjModel
from modules.lossFunc import MultiViewLossFunc
from utils import *
from utils.modelUtils import initialize_optimizer, set_lr_forHand, set_lr_forObj

from absl import flags
from absl import app
from absl import logging

from config import *
import time
import json



## FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230822', 'target db name')   ## name ,default, help
flags.DEFINE_string('seq', '230822_S01_obj_01_grasp_13', 'target sequence name')
flags.DEFINE_string('objClass', 'banana', 'target object name')
FLAGS(sys.argv)


anno_path = "./dataset/hand_db_sample.json"

with open(anno_path, 'r') as file:
    anno = json.loads(file.read())

### update annotation
# anno[annotations], anno[Mesh]
anno['annotations'][0]['id'] = 0


### save full annotation
with open(anno_path, 'w', encoding='utf-8') as file:
    json.dump(anno, file, indent='\t')

