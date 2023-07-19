import os
import sys

sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models/slim'))

# segmentation
from utils import inferenceUtils as infUti
from eval import evalSeg
from HOdatasets.commonDS import *
import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from absl import flags
from absl import app

# preprocess
import mediapipe as mp
import numpy as np

# optimization
from modules.utils.loadParameters import LoadCameraMatrix, LoadDistortionParam
from renderer_v2_pytorch import optimizer_torch


###ADDED LIBRARY MINJAE####
##########MINJAE###########
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import modules.config as cfg
import torch
from pytorch3d.renderer import (
    PerspectiveCameras, FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, PointLights, TexturesVertex,
)
import matplotlib.image as mpimg
import math
###########################
###########END#############


# others
import cv2
import time
import multiprocessing as mlp
import json
import pickle
import cop
import tqdm

import pyrender
import trimesh

### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230612', 'target db Name')   ## name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
flags.DEFINE_string('camID', 'mas', 'main target camera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)


### Config ###
baseDir = os.path.join(os.getcwd(), 'dataset')
handResultDir = os.path.join(baseDir, FLAGS.db) + '_hand'

rootDir = os.path.join(baseDir, FLAGS.db)
lenDBTotal = len(os.listdir(rootDir))

## TODO
"""
    - Issues in camResultDir 
    - segmentation process(original) requires cam_mas_intrinsics.txt ... etc.
        - check HOnnotate_refine.HOdatasets.ho3d_multicamera.dataset.py
        - convert the process to utilize our format (230612_cameraInfo.txt)
"""
camResultDir = os.path.join(baseDir, FLAGS.db) + '_cam'


SEG_CKPT_DIR = 'HOnnotate_refine/checkpoints/Deeplab_seg'
YCB_MODELS_DIR = './HOnnotate_OXR/HOnnotate_refine/YCB_Models/models/'
YCB_OBJECT_CORNERS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../objCorners')
MANO_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../optimization/mano/models/MANO_RIGHT.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
dataset_mix = infUti.datasetMix.OXR_MULTICAMERA
numConsThreads = 1
w = 640
h = 480

### Manual Flags (remove after debug) ###
flag_preprocess = False
flag_segmentation = False
flag_MVobj = True
flag_MVboth = True


def handOptim(seqName, camParamList, metaList, imgList, flag_multi=False):

    intrinsics, extrinsics, distCoeffs = camParamList

    optm = optimizer_torch(camIDset, [intrinsics, extrinsics], flag_multi=flag_multi)

    # run each frame with set of metas/images
    for metas, imgs in zip(metaList, imgList):
        rgbSet = []
        segSet = []
        depthSet = []
        camSet = []
        for name in imgs:
            camID = name[:-5]
            rgb = os.path.join(rootDir, seqName, 'rgb_crop', camID, name+'.png')
            depth = os.path.join(rootDir, seqName, 'depth_crop', camID, name+'.png')
            seg = os.path.join(rootDir, seqName, 'segmentation', camID, 'raw_seg_results', name+'.png')
            rgbSet.append(rgb)
            segSet.append(seg)
            depthSet.append(depth)

        for camID in camIDset:
            camSet.append([intrinsics[camID], extrinsics[camID]])

        data = [camSet, rgbSet, segSet, depthSet, metas]

        if not flag_multi:
            # this is executed now
            optm.run(data)
        else:
            optm.run_multiview(data)

        print("break for debug")
        break

    # dump optimized mano parameters as pkl
    return None


def getCam():
    with open(os.path.join(camResultDir, "cameraParamsBA.json")) as json_file:
        cameraParams = json.load(json_file)
        cameraParams = {camera: np.array(cameraParams[camera]) for camera in cameraParams}

    camInfoName = FLAGS.db + '_cameraInfo.txt'
    intrinsicMatrices = LoadCameraMatrix(os.path.join(camResultDir, camInfoName))
    distCoeffs = {}
    distCoeffs["mas"] = LoadDistortionParam(os.path.join(camResultDir, "mas_intrinsic.json"))
    distCoeffs["sub1"] = LoadDistortionParam(os.path.join(camResultDir, "sub1_intrinsic.json"))
    distCoeffs["sub2"] = LoadDistortionParam(os.path.join(camResultDir, "sub2_intrinsic.json"))
    distCoeffs["sub3"] = LoadDistortionParam(os.path.join(camResultDir, "sub3_intrinsic.json"))

    ### no translation in mano space ###
    for camID in camIDset:
        cameraParams[camID] = cameraParams[camID].reshape(3, 4)

    camParamList = intrinsicMatrices, cameraParams, distCoeffs
    return camParamList

def getMeta(files, seq):
    finalPklDataList = []
    ind = 0

    while (ind < len(files)):
        file = files[ind]
        resultsDir = os.path.join(baseDir, FLAGS.db)

        with open(os.path.join(resultsDir, seq, 'meta', file), 'rb') as f:
            pklData = pickle.load(f)
            finalPklDataList.append(pklData)
        ind = ind + 1

    return finalPklDataList

def read_file(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove newline character at the end of each line
            line = line.strip()
            lines.append(line)
    return lines

def read_obj(file_path):
    verts = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):
                # Parse vertex coordinates
                vertex = line[2:].split()
                vertex = [float(coord) for coord in vertex]
                verts.append(vertex)
            elif line.startswith('f '):
                # Parse face indices
                face = line[2:].split()
                face = [int(index.split('/')[0]) - 1 for index in face]
                faces.append(face)

    obj_data = {'verts': verts, 'faces': faces}
    return obj_data

def apply_transform(matrix, points):
    # Append 1 to each coordinate to convert them to homogeneous coordinates
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply matrix multiplication
    transformed_points = np.dot(matrix, homogeneous_points.T).T

    # Convert back to Cartesian coordinates
    transformed_points_cartesian = transformed_points[:, :3] / transformed_points[:, 3:]

    return transformed_points_cartesian

##### 중요! Tracking 할때 Camera Extrinsic 고려하지 않고 labeling ##########

def display_Obj_scene(idx, mesh_path, intrinsic, extrinsics, pose, img_path):

    # 
    # Load the mesh from an OBJ file
    # Camera extrinsic 적용 X -> camera 좌표가 0,0,0

    verts, faces, _ = load_obj(mesh_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose = np.array(pose)

    pose[0] *= -1
    pose[1] *= -1

    #pose = np.linalg.inv(pose)

    verts = verts.view( -1 , 3)
    verts = verts.numpy()
    
    ### MANUAL INITIAL REFINEMENT CODES ####

    # cam_rot = [25,60,30]

    # cam_rot_rad = [math.radians(rot_deg) for rot_deg in cam_rot]

    # x_rad = cam_rot_rad[0]
    # y_rad = cam_rot_rad[1]
    # z_rad = cam_rot_rad[2]

    # rot_z = np.identity(4)

    # rot_z[0,0] = math.cos(z_rad)
    # rot_z[0,1] = -math.sin(z_rad)
    # rot_z[1,0] = math.sin(z_rad)
    # rot_z[1,1] = math.cos(z_rad)

    # rot_x = np.identity(4)

    # rot_x[1,1] = math.cos(x_rad)
    # rot_x[1,2] = -math.sin(x_rad)
    # rot_x[2,1] = math.sin(x_rad)
    # rot_x[2,2] = math.cos(x_rad)

    # rot_y = np.identity(4)

    # rot_y[0,0] = math.cos(y_rad)
    # rot_y[0,2] = math.sin(y_rad)
    # rot_y[2,0] = -math.sin(y_rad)
    # rot_y[2,2] = math.cos(y_rad)

    # # xform = rot_y*rot_x*rot_z
    # xform = np.dot(rot_y, np.dot(rot_x, rot_z))

    # xform[0,3] = 0
    # xform[1,3] = 0.1
    # xform[2,3] = 0

    # #verts = apply_transform(xform,verts)
    # #verts = apply_transform(pose,verts)

    # xform2 = np.identity(4)
    # xform2[0,3] = -0.07
    # xform2[1,3] = 0.045
    # xform2[2,3] = 0
    # #verts = apply_transform(xform2,verts)

    #pose_matrix = np.dot(xform2, np.dot(pose, xform))

    verts = apply_transform(pose,verts)
    verts = torch.FloatTensor(verts)

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    meshes = Meshes(verts=[verts], faces=[faces.verts_idx],textures=textures).to(device)

    intrinsic = np.array(intrinsic)
    
    image_size = (cfg.ORIGIN_HEIGHT, cfg.ORIGIN_WIDTH)

    focal_l = (intrinsic[0, 0], intrinsic[1, 1])
    principal_p = (intrinsic[0, -1], intrinsic[1, -1])

    cameras = PerspectiveCameras(device=device, image_size=(image_size,), focal_length=(focal_l,),
                                 principal_point=(principal_p,), in_ndc=False)
    
    # Set blend params
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Create a point light source
    #lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))

    # Create a rasterization settings object
    raster_settings = RasterizationSettings(
        image_size=(cfg.ORIGIN_HEIGHT, cfg.ORIGIN_WIDTH),
        blur_radius=0, #np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=1,
        bin_size = -1,
        max_faces_per_bin = None
    )

    # Create a mesh renderer with a soft Phong shader
    renderer_rgb = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
        )
    )

    rasterizer_depth = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    
    # Render the mesh
    images = renderer_rgb(meshes)

    # Convert the rendered image to a numpy array
    image_np = images[0, ..., :3].detach().cpu().numpy()

    # Display the rendered image
    img_rgb = mpimg.imread(img_path)[:,:,:3]

    plt.imsave('./vis/output'+str(idx)+'.png', (image_np + img_rgb) / 2 )



def main(argv):
    ### Setup ###

    ### Multi-view object pose optimization ###
    '''
    [TODO]
        - Load object pose & mesh (currently ICG)
        - render the scene (object only, utilize pytorch3D)
        - compute depth map error, silhouette error for each view
        - optimize object initial pose per frame (torch-based)
    '''

    if flag_MVobj:
        print("TODO")


    ### Multi-view hand-object pose optimization ###
    '''
    [Current]
        - optimization.NIA_handPoseMultiview.py
        - tensorflow_v1 based optimization (without depth rendering) 
        - only minimizing losses between 'mano mesh pose - multiview hand pose'
    
    [TODO]
        - torch based optimization
        - pytorch3D rendering (TW.H)
        - adopt losses from HOnnotate (optimization.handPoseMultiframe.py)    
    '''
    if flag_MVboth:
        # load camera param
        camParamList = getCam()     # intrinsicMatrices, cameraParams(R, T), distCoeffs

        # load meta data(hand results)
        for seqIdx, seqName in enumerate(sorted(os.listdir(rootDir))):
            pklDataperCam = []      # [[mas_pkl_list], [sub1_pkl_list], ...]
            imgNameperCam = []
            for camID in camIDset:
                pklFilesList = os.listdir(os.path.join(rootDir, seqName, 'meta', camID))
                frameList = [ff[:-4] for ff in pklFilesList if 'pkl' in ff]
                frameList = sorted(frameList)
                imgNameperCam.append(frameList)

                pklFilesList = [camID + '/' + f + '.pkl' for f in frameList]
                pklDataList = getMeta(pklFilesList, seqName)
                pklDataperCam.append(pklDataList)

            metas = []
            imgs = []
            frameLen = len(imgNameperCam[0])
            for i in range(frameLen):
                metasperFrame = []
                imgsperFrame = []
                for camIdx in range(len(camIDset)):
                    meta = pklDataperCam[camIdx][i]
                    img = imgNameperCam[camIdx][i]
                    metasperFrame.append(meta)
                    imgsperFrame.append(img)

                metas.append(metasperFrame)
                imgs.append(imgsperFrame)
            

            ######### #Single-View / Mocap 사용 / Object pose Rendering with pytorch3D #########
            if seqName == '230612_mustard' :
                
                img_dir = '/minjay/HOnnotate_OXR/dataset/230612/'

                cam_idx = {'mas':0,'sub1':1,'sub2':2,'sub3':3}
                obj_Name = seqName.split('_')[1]

                print(obj_Name)
                
                scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

                intrinsics, extrinsics, distCoeffs = camParamList
                
                pose_file_path = '/minjay/HOnnotate_OXR/ObjPose/' + seqName + '.txt'
                lines = read_file(pose_file_path)

                view_point_cam = lines[0]

                view_point_cam = intrinsics[view_point_cam]
                
                obj_file_path = '/minjay/HOnnotate_OXR/ObjTemplate/' + obj_Name + '.obj'
                obj_data = read_obj(obj_file_path)
                
                for idx, img in enumerate(imgs) :
                    #img_path = os.path.join(img_dir,seqName,'rgb_crop',lines[0],img[cam_idx[lines[0]]]+'.png')
                    #/minjay/HOnnotate_OXR/dataset/230612/230612_baseball/rgb/mas_1.png
                    img_path = os.path.join(img_dir,seqName,'rgb',lines[0]+'_'+str(idx)+'.png')

                    if idx < 1 :
                        # pose 가 2번째 frame 부터 시작 함
                        continue 

                    else :
                        print('idx',idx)
                        pose = lines[idx*2].split(',')
                        pose = [ float(x) for x in pose ] 
                        pose_4_4 = [ pose[x:x+4] for x in range(0, len(pose), 4) ] 
                        extrinsics_ = extrinsics[lines[0]]
                        
                        display_Obj_scene(idx, obj_file_path, view_point_cam, extrinsics_, pose_4_4, img_path)
                                         #(idx, mesh_path, intrinsic, extrinsics, pose, img_path):
                
            else :
                continue
            
            ######### #Single-View / Mocap 사용 / Object pose Rendering with pytorch3D #########


            handOptim(seqName, camParamList, metas, imgs, flag_multi=False)

    ### Multi-frame pose refinement ###
    '''
    [TODO]
        - optimization over multi-frames while including every hand-object loss        
        - check optimization.handObjectRefinementMultiframe.py
    '''




    print("end")



if __name__ == '__main__':
    app.run(main)







