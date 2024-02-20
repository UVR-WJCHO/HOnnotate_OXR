import pandas as pd
import os
import tqdm

# Replace 'your_directory_path' with the path to your directory
directory_path = 'csv_output'
filter_thresh  = 0.5

# Dictionary to store df
df = {}

camera = ['mas','sub1','sub2','sub3']

origin_num = 0
filtered_num = 0
dfs = []
# Loop through each file in the directory
for idx, filename in enumerate(tqdm.tqdm(os.listdir(directory_path))):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # print(f'read file {file_path}')
        # Read the CSV file and store it in the dictionary
        df[filename] = pd.read_csv(file_path)
        origin_num += len(df[filename])
        for index, row in df[filename].iterrows():
            
            #print(f"Index: {index}, Trial: {row['Trial']}, Seq: {row['Seq']}, Value: {row['Value']}")

            if float(row['Value']) >= filter_thresh :

                #print(f"Index: {index}, Trial: {row['Trial']}, Seq: {row['Seq']}, Value: {row['Value']}")

                trial_temp = row['Trial']
                seq_temp = row['Seq'].split('_')[1]

                if index in df[filename].index:
                    df[filename] = df[filename].drop(index)

                    drop_dix = []
                    for cam in camera :
                        row_indices = df[filename].index[ (df[filename]['Trial'] == trial_temp) & (df[filename]['Seq'] == f'{cam}_{seq_temp}') ].tolist()

                        if row_indices != [] :
                            df[filename] = df[filename].drop(row_indices[0])
  
                
        df[filename].reset_index(drop=True)
        df[filename].insert(0, 'Sequence', filename[:-4])
        dfs.append(df[filename])
        filtered_num += len(df[filename])

merged_df = pd.concat(dfs)
merged_df = merged_df.rename(columns={'Seq': 'Frame'})
filename_lst = filename.split('.')
merged_df.to_csv(f'{directory_path}_filtered.csv', index=False)

print("origin num : ", origin_num)
print("filtered num : ", filtered_num)
