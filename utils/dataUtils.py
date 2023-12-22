import os
import sys
import torch

import numpy as np
import math
import json
from config import *

def save_annotation(targetDir, trialName, frame, seq, pred, pred_obj, side):
    #seq ='230822_S01_obj_01_grasp_13'
    db = seq.split('_')[0]
    subject_id = seq.split('_')[1][1:]
    obj_id = seq.split('_')[3]
    grasp_id = seq.split('_')[5]
    trial_num = trialName.split('_')[1]
    cam_list = ['mas', 'sub1', 'sub2', 'sub3']
    anno_base_path = os.path.join(targetDir, trialName, 'annotation')
    anno_path_list = []
    for camID in cam_list:
        anno_path_list.append(os.path.join(anno_base_path, camID ,f'anno_{frame:04}.json'))

    for anno_path in anno_path_list:
        anno = None
        ### load current annotation(include updated meta info.)
        with open(anno_path, 'r', encoding='cp949') as file:
            anno = json.load(file)
        imgID = anno['images']['id']
        ### update annotation
        # anno[annotations], anno[Mesh]
        anno['annotations'][0]['id'] = str(db) + str(subject_id) + str(obj_id) + str(grasp_id) + str(trial_num) + str(frame)
        anno['annotations'][0]['image_id'] = imgID
        anno['annotations'][0]['class_id'] = grasp_id
        anno['annotations'][0]['class_name'] = GRASPType(int(grasp_id)).name
        anno['annotations'][0]['type'] = "K"
        anno['annotations'][0]['data'] = pred['joints'].tolist()
        anno['Mesh'][0]['id'] = str(db) + str(subject_id) + str(obj_id) + str(grasp_id) + str(trial_num) + str(frame)
        anno['Mesh'][0]['image_id'] = imgID
        anno['Mesh'][0]['class_id'] = grasp_id
        anno['Mesh'][0]['class_name'] = GRASPType(int(grasp_id)).name
        anno['Mesh'][0]['object_name'] = OBJType(int(obj_id)).name
        obj_files = []
        obj_mat = []
        for key in pred_obj:
            obj_files.append(pred_obj[key][1])
            obj_mat.append(pred_obj[key][0])

        anno['Mesh'][0]['object_file'] = obj_files
        anno['Mesh'][0]['object_mat'] = obj_mat
        anno['Mesh'][0]['mano_side'] = side
        anno['Mesh'][0]['mano_trans'] = pred['rot'].tolist()
        anno['Mesh'][0]['mano_pose'] = pred['pose'].tolist()
        anno['Mesh'][0]['mano_betas'] = pred['shape'].tolist()

        # post-process contact map
        contact_map = pred['contact']
        if contact_map is not None:
            contact_idx = torch.where(contact_map > 0)
            if not contact_idx[0].nelement() == 0:
                max = contact_map[contact_idx].max()
                contact_map[contact_idx] = contact_map[contact_idx] / max
                contact_map[contact_idx] = 1 - contact_map[contact_idx]
            contact_map[contact_map == -1.] = 0.
            contact_map = contact_map.tolist()

        anno['Mesh'][0]['contact'] = contact_map

        ### save full annotation
        with open(anno_path, 'w', encoding='cp949') as file:
            json.dump(anno, file, indent='\t', ensure_ascii=False)


def generate_pose( rot, trans ) :

    cam_rot = rot

    cam_rot_rad = [math.radians(rot_deg) for rot_deg in cam_rot]

    x_rad = cam_rot_rad[0]
    y_rad = cam_rot_rad[1]
    z_rad = cam_rot_rad[2]

    rot_z = np.identity(4)

    rot_z[0,0] = math.cos(z_rad)
    rot_z[0,1] = -math.sin(z_rad)
    rot_z[1,0] = math.sin(z_rad)
    rot_z[1,1] = math.cos(z_rad)

    rot_x = np.identity(4)

    rot_x[1,1] = math.cos(x_rad)
    rot_x[1,2] = -math.sin(x_rad)
    rot_x[2,1] = math.sin(x_rad)
    rot_x[2,2] = math.cos(x_rad)

    rot_y = np.identity(4)

    rot_y[0,0] = math.cos(y_rad)
    rot_y[0,2] = math.sin(y_rad)
    rot_y[2,0] = -math.sin(y_rad)
    rot_y[2,2] = math.cos(y_rad)

    # xform = rot_y*rot_x*rot_z
    xform = np.dot(rot_y, np.dot(rot_x, rot_z))

    xform2 = np.identity(4)
    xform2[0,3] = trans[0]
    xform2[1,3] = trans[1]
    xform2[2,3] = trans[2]
    #verts = apply_transform(xform2,verts)

    pose_matrix = np.dot(xform2, xform)

    return pose_matrix

def apply_transform(matrix, points):
    # Append 1 to each coordinate to convert them to homogeneous coordinates
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply matrix multiplication
    transformed_points = np.dot(matrix, homogeneous_points.T).T

    # Convert back to Cartesian coordinates
    transformed_points_cartesian = transformed_points[:, :3] / transformed_points[:, 3:]

    return transformed_points_cartesian
