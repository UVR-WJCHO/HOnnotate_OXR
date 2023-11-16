import os
import pandas as pd


target_path = "./230919_S24_obj_04_grasp_11"
trial_list = os.listdir(target_path)
print(trial_list)

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

for trial_path in trial_list:
    log_path = os.path.join(target_path, trial_path, 'log')
    log_list = os.listdir(log_path)
    print(log_list)

    for csv_file in log_list:
        log_name = csv_file.split('.')[0]
        print(log_name)
        log_df = pd.read_csv(os.path.join(log_path, csv_file))
        avg_value_each_csv = float(log_df.columns[-1])

        log_dict[log_name].append(avg_value_each_csv)

average_dict = {}
for key, values in log_dict.items():
    if values:
        average_dict[key] = sum(values) / len(values)
    else:
        average_dict[key] = None
average_df = pd.DataFrame(list(average_dict.items()), columns=['Metric', 'Average'])

save_path = os.path.join(target_path, 'log_averages.csv')
average_df.to_csv(save_path, index=False)