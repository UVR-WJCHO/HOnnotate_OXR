import os
import sys
import pickle
from absl import flags
from absl import app
import json
from natsort import natsorted
import pandas as pd


### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'test_result', 'target db Name')   ## name ,default, help
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

baseDir = os.path.join(os.getcwd(), 'dataset')

## kpts_{} will be changed to tip_{}
log_name_list = ['depth_f1',
                 'depth_precision',
                 'depth_recall',
                 'kpts_f1',
                 'kpts_precision',
                 'kpts_recall',
                 'mesh_f1',
                 'mesh_precision',
                 'mesh_recall']

log_dict = {key: [] for key in log_name_list}


def load_avg(targetDir, seq, trial):
    log_base_path = os.path.join(targetDir, seq, trial, 'log')
    # log_list = os.listdir(log_base_path)

    for log_name in log_name_list:
        log_path = os.path.join(log_base_path, log_name+'.csv')
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            avg_value_each_csv = float(log_df.columns[-1])
            log_dict[log_name].append(avg_value_each_csv)
    return True


def main():
    rootDir = os.path.join(baseDir, FLAGS.db)
    seq_list = natsorted(os.listdir(rootDir))

    for seqIdx, seqName in enumerate(seq_list):
        seqDir = os.path.join(rootDir, seqName)
        for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
            load_avg(rootDir, seqName, trialName)

    average_dict = {}
    for key, values in log_dict.items():
        if values:
            average_dict[key] = sum(values) / len(values)
        else:
            average_dict[key] = None
    average_df = pd.DataFrame(list(average_dict.items()), columns=['Metric', 'Average'])
    output_name = FLAGS.db + '_evaluation_average.csv'
    save_path = os.path.join(baseDir, output_name)
    average_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()