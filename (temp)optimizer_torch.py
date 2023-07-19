import os
import sys
import math
import cv2
import argparse
import numpy as np
import torch
from collections import OrderedDict
import tkinter as tk
from PIL import Image, ImageTk

from modules.HandsForAll import mano_wrapper
from modules.HandsForAll.utils import *
import modules.HandsForAll.params as param

from modules.HandsForAll.kpts_global_projector import kpts_global_projector
# temp

MANO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modules/manopth/mano/models')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class pose_annotator:
    def __init__(self, args=param):
        self.args = args
        self.init_kpts_projector()
        self.init_variables()
        self.img_display_np = np.ones((params.IMG_SIZE + params.IMG_PADDING * 2, \
                                      (params.IMG_SIZE + params.IMG_PADDING * 2) * 3, 3), dtype=np.uint8) * 128
        self.init_model_3d_3rd()
        self.hand_mode = 'r'

    def run(self, camSet, rgbSet, depthSet, metas):

        # debug main cam first
        intrinsic, extrinsic = camSet[0]

        intrinsic[0, -1] /= (1920. / 640.)
        intrinsic[1, -1] /= (1080. / 480.)

        rgb = rgbSet[0]
        meta = metas[0]

        self.image_target = rgb
        print('Annotating {}'.format(self.image_target))
        self.init_frame_info()

        self.mano_fit_tool.reset_parameters(keep_mano=True)
        # self.mano_fit_tool.mano_model.set_cam_intrinsic(K=intrinsic)


        ### our results ###
        # # kpts[idx] = np.array([h, w]), check order
        # kpts_3d_can_pred_np = np.copy(meta['kpts_crop'])    # np(21, 3)     -1~1
        # kpts_3d_can_pred_np[:, 0] = (kpts_3d_can_pred_np[:, 0] / (params.IMG_HEIGHT / 2.0)) - 1.0
        # kpts_3d_can_pred_np[:, 1] = (kpts_3d_can_pred_np[:, 1] / (params.IMG_WIDTH / 2.0)) - 1.0
        # kpts_3d_can_pred_np[:, 2] = (kpts_3d_can_pred_np[:, 2] - min(kpts_3d_can_pred_np[:, 2])) \
        #                             / float(params.DEPTH_WINDOW)
        #kpts_2d_glob_gt_np = np.copy(meta['kpts_crop'][:, :2])



        ### original model ###
        kpts_2d_glob_gt_np = np.copy(meta['kpts_crop'][:, :2])[:, [1, 0]]
        kpts_2d_glob_gt_norm = np.copy(kpts_2d_glob_gt_np)
        kpts_2d_glob_gt_norm[:, 0] /= params.IMG_HEIGHT
        kpts_2d_glob_gt_norm[:, 1] /= params.IMG_WIDTH

        kpts_2d = np.copy(meta['kpts_crop'][:, :2])[:, [1, 0]]

        # debug = paint_kpts(rgb, img=None, kpts=kpts_2d)
        # cv2.imshow("debug", debug)
        # cv2.waitKey(0)

        kpts_2d = kpts_2d / params.IMG_SIZE * params.CROP_SIZE_PRED
        heatmaps_np = generate_heatmaps((params.CROP_SIZE_PRED, params.CROP_SIZE_PRED), \
                                        params.CROP_STRIDE_PRED, kpts_2d, sigma=params.HEATMAP_SIGMA, is_ratio=False)
        heatmaps_tensor = torch.from_numpy(heatmaps_np.transpose(2, 0, 1)).unsqueeze_(0).cuda()
        kpts_3d_can_pred = self.model_3d_3rd(heatmaps_tensor)
        # Pred 3D canonical pose
        kpts_3d_can_pred_np = kpts_3d_can_pred.cpu().data.numpy()[0].reshape(21, 3)


        # Fit kpts 3d canonical
        self.mano_fit_tool.fit_3d_can_init(kpts_3d_can_pred_np, is_tracking=False)
        # Fit xyz root
        ################### check below function ####################
        kpts_3d_glob_projected = self.kpts_global_project_tool.canon_to_global \
            (kpts_2d_glob_gt_norm, kpts_3d_can_pred_np)
        self.mano_fit_tool.set_xyz_root_with_projection(kpts_3d_glob_projected)

        self.mano_fit_tool.fit_xyz_root_init(kpts_2d_glob_gt_np, is_tracking=False)
        # Fit pose
        self.mano_fit_tool.fit_all_pose(kpts_3d_can_pred_np, kpts_2d_glob_gt_np, is_tracking=False)

        self.img_center = self.get_rendered_img()
        self.update_img_display("fit results")


    def init_model_3d_3rd(self):
        from modules.HandsForAll.models.resnet import resnet_pose
        model = resnet_pose.resnet9(num_classes=params.NUM_KPTS * 3, num_inchan=params.NUM_KPTS)
        pretrained_3d_path = params.MODEL_3D_3RD_PATH
        if not os.path.exists(pretrained_3d_path):
            self.model_3d_3rd = None
            print('Model third-person 3D not found : {}'.format(pretrained_3d_path))
        else:
            print('Loading {}'.format(pretrained_3d_path))
            print('model_3d third-person #params: {}'.format(sum(p.numel() for p in model.parameters())))
            state_dict = torch.load(pretrained_3d_path)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            self.model_3d_3rd = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            self.model_3d_3rd.eval()
            print('Model third-person 3D succesfully loaded')

    def init_kpts_projector(self):
        print("check foc_l, currently assume fx=fy")
        self.kpts_global_project_tool = kpts_global_projector('mano_hand', foc_l=908.0, cam_h=480, cam_w=640)

    def init_variables(self):
        self.mano_fit_tool = mano_wrapper.mano_fitter(MANO_ROOT, is_annotating=True)
        self.mano_fit_tool.set_input_size(params.IMG_HEIGHT, params.IMG_WIDTH)
        self.img_left, self.img_center, self.img_right = None, None, None
        self.crop_center = None
        self.crop_size = params.CROP_SIZE_DEFAULT
        self.crop_box = None
        self.crop_info = None
        self.is_debugging = False
        self.img_side_is_left = None
        self.joint_selected = 0
        self.joint_anno_dict_l = dict()
        self.joint_anno_dict_r = dict()
        self.img_toggle_state = False
        self.mano_info_loaded = None
        self.kpt_2d_load_list = [i for i in range(21)]
        self.results_saved = False

    def init_frame_info(self):
        # Init display images
        self.img_orig, self.img_left = load_img_mano(self.image_target)
        if self.hand_mode == 'l':
            self.img_left = cv2.flip(self.img_left, 1)
        self.img_right = np.zeros_like(self.img_left)

        self.img_center = self.get_rendered_img()
        # self.update_img_display("init")

    # Utility functions
    def get_rendered_img(self):
        return self.mano_fit_tool.get_rendered_img()

    def update_img_display(self, msg):
        # Display left image
        row_s_l = params.IMG_PADDING
        row_e_l = row_s_l + params.IMG_HEIGHT
        col_s_l = params.IMG_PADDING
        col_e_l = col_s_l + params.IMG_WIDTH

        img_left_display = self.img_left.copy()
        self.img_display_np[row_s_l:row_e_l, col_s_l:col_e_l, :] = img_left_display

        # Display center image
        row_s_c = row_s_l
        row_e_c = row_e_l
        col_s_c = col_e_l + params.IMG_PADDING * 2
        col_e_c = col_s_c + params.IMG_WIDTH

        self.img_display_np[row_s_c:row_e_c, col_s_c:col_e_c, :] = self.img_center

        # Display right image
        row_s_r = row_s_l
        row_e_r = row_e_l
        col_s_r = col_e_c + params.IMG_PADDING * 2
        col_e_r = col_s_r + params.IMG_WIDTH

        # Display blended image
        img_1 = np.copy(img_left_display)
        img_2 = np.copy(self.img_center)
        img_3 = cv2.addWeighted(img_1, 0.5, img_2, 0.7, 0)

        img_right_display = img_3.copy()
        self.img_display_np[row_s_r:row_e_r, col_s_r:col_e_r, :] = img_right_display
        print("call update_img_display()")
        cv2.imshow(msg, self.img_display_np)
        cv2.waitKey(0)

