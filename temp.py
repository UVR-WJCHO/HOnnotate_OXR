import os
import pickle
from natsort import natsorted



rootDir = "C:/Projects/OXR_projects/HOnnotate_OXR/dataset/230922_obj"
seq_list = natsorted(os.listdir(rootDir))
for seqIdx, seqName in enumerate(seq_list):
    if seqIdx < 4:
        continue
    if seqIdx == 34:
        break
    seqDir = os.path.join(rootDir, seqName)
    files = os.listdir(seqDir)
    files_scale = [file for file in files if file.endswith("obj_scale.pkl")]

    file_scale = files_scale[0]

    scale_path = os.path.join(seqDir, file_scale)
    with open(scale_path, 'rb') as f:
        scale = pickle.load(f)

    print(scale)