import pandas as pd

file_path = 'csv_output_filtered.csv'
df = pd.read_csv(file_path)

df[['Subject', 'Object', 'Grasp']] = df['Sequence'].str.split('_', expand=True)[[1, 3, 5]]
subject_counts = df['Subject'].value_counts()
object_counts = df['Object'].value_counts()
grasp_counts = df['Grasp'].value_counts()
object_grasp_counts = df.groupby(['Object', 'Grasp']).size()

print(subject_counts)
print(object_counts)
print(grasp_counts)
print(object_grasp_counts)

subject_counts.to_csv('subject_counts.csv')

object_counts.to_csv('object_counts.csv')

grasp_counts.to_csv('grasp_counts.csv')

object_grasp_counts_df = object_grasp_counts.reset_index()
object_grasp_counts_df.columns = ['Object', 'Grasp', 'Count']
object_grasp_counts_df.to_csv('object_grasp_counts.csv', index=False)