import os
import sys
import pickle
from absl import flags
from absl import app
import json
from natsort import natsorted
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
DATA STRUCTURE

tool_TTA_evaluation_final.py
data
    NIA_output
        HAND_GESTURE
            {db}                    ## ex. 20231207
                label_data
                    230905_S01_obj_16_grasp_14
                    230905_S01_obj_16_grasp_27
                    ...
                
"""


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '20231207', 'target db Name')   ## name ,default, help
flags.DEFINE_string('db_output', 'label_data', 'target db Name')   ## name ,default, help
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'data', 'NIA_output', 'HAND_GESTURE')

## kpts_{} will be changed to tip_{}
log_names = ['depth', 'mesh', 'kpts']

log_name_list_all = ['depth_f1', 'mesh_f1', 'kpts_f1']

log_name_list_tip = ['tip_f1']

log_dict = {key: [] for key in log_name_list_all}
log_dict_tip = {key: [] for key in log_name_list_tip}

def load_avg(targetDir, seq, trial, valid_dict):
    log_base_path = os.path.join(targetDir, seq, trial, 'log')

    for log_name in log_names:
        log_path_f1 = os.path.join(log_base_path, log_name + '_f1'+'.csv')

        if os.path.exists(log_path_f1):
            value_list_f1 = []

            log_df_f1 = pd.read_csv(log_path_f1, skiprows=2, skip_blank_lines=True)
            if 'frame' not in log_df_f1:
                log_df_f1 = pd.read_csv(log_path_f1, skiprows=1, skip_blank_lines=True)

            log_frame_f1 = log_df_f1['frame']

            for key, values in valid_dict.items():
                for value in values:
                    if value in log_frame_f1.values:
                        right_idx = log_frame_f1[log_frame_f1 == value].index[0]

                        valid_log_f1 = float(log_df_f1[key][right_idx])

                        if isinstance(valid_log_f1, float) and valid_log_f1 > 0.001:
                            value_list_f1.append(float(valid_log_f1))

            if not len(value_list_f1) == 0:
                avg_value_f1 = np.average(value_list_f1)
                log_dict[log_name+'_f1'].append(avg_value_f1)

    for log_name in log_name_list_tip:
        log_path = os.path.join(log_base_path, log_name + '.csv')
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            avg_value_each_csv = float(log_df.columns[-1])
            log_dict_tip[log_name].append(avg_value_each_csv)

    return True


def extract_valid(targetDir, seq, trial):
    anno_base_path = os.path.join(targetDir, seq, trial, 'annotation')
    valid_dict = {}
    for camName in os.listdir(anno_base_path):
        valid_dict[camName] = []
        annoDir = os.path.join(anno_base_path, camName)
        annoList = natsorted(os.listdir(annoDir))

        for anno in annoList:
            frame = int(anno.split('_')[1][:-5])
            valid_dict[camName].append(frame)

    return valid_dict


def main():
    rootDir = os.path.join(baseDir, FLAGS.db, FLAGS.db_output)

    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(tqdm(seq_list)):
        seqDir = os.path.join(rootDir, seqName)
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            valid_dict = extract_valid(rootDir, seqName, trialName)
            load_avg(rootDir, seqName, trialName, valid_dict)

    ## save average value of metrics
    average_dict = {}
    for key, values in log_dict.items():
        if values:
            average_dict[key] = sum(values) / len(values)
        else:
            average_dict[key] = None
    average_df = pd.DataFrame(list(average_dict.items()), columns=['Metric', 'Average'])
    output_name = FLAGS.db + '_evaluation_average.csv'
    save_path = os.path.join(os.getcwd(), output_name)
    average_df.to_csv(save_path, index=False)

    output_raw_name = FLAGS.db + '_evaluation_all.json'
    save_path = os.path.join(os.getcwd(), output_raw_name)
    with open(save_path, 'w', encoding='UTF-8 SIG') as file:
        json.dump(log_dict, file, indent='\t', ensure_ascii=False)

    ## save average value of tip metrics
    average_dict = {}
    for key, values in log_dict_tip.items():
        if values:
            average_dict[key] = sum(values) / len(values)
        else:
            average_dict[key] = None
    average_df = pd.DataFrame(list(average_dict.items()), columns=['Metric', 'Average'])
    output_name = FLAGS.db + '_evaluation_average_tip.csv'
    save_path = os.path.join(os.getcwd(), output_name)
    average_df.to_csv(save_path, index=False)

    output_raw_name = FLAGS.db + '_evaluation_all_tip.json'
    save_path = os.path.join(os.getcwd(), output_raw_name)
    with open(save_path, 'w', encoding='UTF-8 SIG') as file:
        json.dump(log_dict_tip, file, indent='\t', ensure_ascii=False)


if __name__ == '__main__':
    main()