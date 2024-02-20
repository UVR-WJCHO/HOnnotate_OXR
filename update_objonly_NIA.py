import os
import sys
import pickle
import warnings
warnings.filterwarnings(action='ignore')

from absl import flags
from absl import app
import json
from natsort import natsorted
import torch
import torch.nn as nn
import numpy as np
from config import *

from manopth.manolayer import ManoLayer
from modules.renderer import Renderer
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from utils.lossUtils import *

from modules.deepLabV3plus.oxr_predict import predict as deepSegPredict
from modules.deepLabV3plus.oxr_predict import load_model as deepSegLoadModel
import pandas as pd
import time
# multiprocessing
import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from absl import logging

from optimizer_NIA import *
from dataloader_NIA import loadNIADB

### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_integer('start', 0, 'start idx of sequence(ordered)')   ## name ,default, help
flags.DEFINE_integer('end', 50, 'end idx of sequence(ordered)')   ## name ,default, help
FLAGS(sys.argv)

camIDset = ['mas', 'sub1', 'sub2', 'sub3']

LOSS_DICT = ['seg_obj', 'depth_obj']

### N5 path / N1 path ###
baseDir = os.path.join('dataset/NIA_db_wj')
# baseDir = os.path.join('/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets')

objModelDir = os.path.join(os.getcwd(), 'obj_scanned_models')
mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
csv_save_path = os.path.join(os.getcwd(), 'csv_output_filtered.csv')
filtered_df = pd.read_csv(csv_save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
#                        center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
#
# hand_faces_template = mano_layer.th_faces.repeat(1, 1, 1)



def update_object_only(db):
    ## Initialize loss function
    loss_func = MultiViewLossFunc_OBJ(device=device, dataloaders=db, renderers=db.renderer_set,
                                      losses=LOSS_DICT).to(device)
    loss_func.set_main_cam(main_cam_idx=0)
    loss_func.set_object_main_extrinsic(0)  # Set object's main camera extrinsic as mas

    model_obj = ObjModel(device, 1, db.obj_mesh_data).to(device)

    for camIdx, camID in enumerate(camIDset):
        if camID in db.valid_cams:
            for idx in range(db.get_len(camID)):
                obj_pose = db.get_obj_pose(camID, idx)
                obj_pose = obj_pose[:-1, :]
                model_obj.update_pose(pose=obj_pose)

                ### initialize optimizer, scheduler
                lr_init_obj = 0.2

                ### update obj
                optimize_obj(model_obj, loss_func, camIdx, idx, lr_init_obj, db.seq, db.trial, target_iter=100, flag_vis=True)

                pred_obj = model_obj()
                pred_obj_anno = [model_obj.get_object_mat().tolist(), db.obj_mesh_name]

                loss_func.visualize(pred_obj=pred_obj, camIdx=camIdx, frame=idx)

                # save_annotation(target_dir_result, trialName, frame, target_seq, pred_obj_anno, CFG_MANO_SIDE)
                print("end %s - frame %s" % (db.trial, idx))


def error_callback(result):
    print("Error!")

def done_callback(result):
    # print("Done. Result: ", result)
    return


def main():
    t1 = time.time()
    logging.get_absl_handler().setFormatter(None)

    base_source = os.path.join(baseDir, 'source')
    base_anno = os.path.join(baseDir, 'annotation')
    # base_source = os.path.join(baseDir, '1_Source_data')
    # base_anno = os.path.join(baseDir, '2_Labeling_data')

    seq_list = natsorted(os.listdir(base_anno))
    print("total sequence # : ", len(seq_list))

    if FLAGS.start != None and FLAGS.end != None:
        seq_list = seq_list[FLAGS.start:FLAGS.end]
        print(f"execute {FLAGS.start} to {FLAGS.end}")

    total_count = 0
    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(base_anno, seqName)

        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            seq_count = 0

            valid_cam = []
            for camID in camIDset:
                p = os.path.join(seqDir, trialName, 'annotation', camID)
                if os.path.exists(p):
                    temp = len(os.listdir(p))
                    seq_count += temp
                    total_count += temp
                    valid_cam.append(camID)

            db = loadNIADB(base_anno, base_source, seqName, trialName, valid_cam, seq_count, device)

            update_object_only(db)


if __name__ == '__main__':
    main()