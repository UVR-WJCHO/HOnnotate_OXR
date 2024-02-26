import os
import sys
import warnings
warnings.filterwarnings(action='ignore')
import json
import pandas as pd

from modules.renderer import Renderer
from pytorch3d.io import load_obj
from utils.lossUtils import *
from modules.deepLabV3plus.oxr_predict import predict as deepSegPredict
from modules.deepLabV3plus.oxr_predict import load_model as deepSegLoadModel


camIDset = ['mas', 'sub1', 'sub2', 'sub3']


objModelDir = os.path.join(os.getcwd(), 'obj_scanned_models')
# csv_save_path = os.path.join(os.getcwd(), 'csv_output_filtered.csv')
# filtered_df = pd.read_csv(csv_save_path)


def extractBbox(hand_2d, image_rows=1080, image_cols=1920, bbox_w=640, bbox_h=480):
        # consider fixed size bbox
        x_min_ = min(hand_2d[:, 0])
        x_max_ = max(hand_2d[:, 0])
        y_min_ = min(hand_2d[:, 1])
        y_max_ = max(hand_2d[:, 1])

        x_avg = (x_min_ + x_max_) / 2
        y_avg = (y_min_ + y_max_) / 2

        x_min = max(0, x_avg - (bbox_w / 2))
        y_min = max(0, y_avg - (bbox_h / 2))

        if (x_min + bbox_w) > image_cols:
            x_min = image_cols - bbox_w
        if (y_min + bbox_h) > image_rows:
            y_min = image_rows - bbox_h

        bbox = [x_min, y_min, bbox_w, bbox_h]
        return bbox, [x_min_, x_max_, y_min_, y_max_]


class deeplab_opts():
    def __init__(self, object_id, device):
        self.model = 'deeplabv3plus_mobilenet'
        self.output_stride = 16
        self.gpu_id = '0'
        self.ckpt = "./modules/deepLabV3plus/checkpoints/%02d_best_deeplabv3plus_mobilenet_oxr_os16.pth" % int(object_id)

        assert os.path.isfile(self.ckpt), "no ckpt files for object %02d" % int(object_id)
        # print("...loading ", self.ckpt)
        self.checkpoint = torch.load(self.ckpt, map_location=device)


class loadNIADB():
    def __init__(self, baseDir, base_anno, base_source, seq, trialName, valid_cams, valid_num, device, csv_list=None):
        self.device = device
        self.baseDir = baseDir
        self.seq = seq  # 230612_S01_obj_01_grasp_1
        self.subject_id = seq.split('_')[1][1:]
        self.obj_id = seq.split('_')[3]
        self.grasp_id = seq.split('_')[5]
        self.trial = trialName
        self.trial_num = trialName.split('_')[1]
        self.valid_num = valid_num
        self.csv_list = csv_list

        ## load each camera parameters ##
        anno_base_path = os.path.join(base_anno, seq, trialName, 'annotation')

        Ks_list = []
        Ms_list = []
        for camID in camIDset:
            if camID in valid_cams:
                anno_list = os.listdir(os.path.join(anno_base_path, camID))
                anno_path = os.path.join(anno_base_path, camID, anno_list[0])
                with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                    anno = json.load(file)

                Ks = torch.FloatTensor(np.squeeze(np.asarray(anno['calibration']['intrinsic']))).to(device)

                Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
                Ms = np.reshape(Ms, (3, 4))
                ## will be processed in postprocess, didn't.
                Ms[:, -1] = Ms[:, -1] / 10.0
                Ms = torch.Tensor(Ms).to(device)

                Ks_list.append(Ks)
                Ms_list.append(Ms)
            else:
                Ks_list.append(None)
                Ms_list.append(None)

        self.Ks_list = Ks_list
        self.Ms_list = Ms_list
        self.valid_cams = valid_cams

        ## set renderer for each cam ##
        default_M = np.eye(4)[:3]
        renderer_set = []
        for camIdx, camID in enumerate(camIDset):
            if camID in valid_cams:
                renderer = Renderer('cuda', 1, default_M, Ks_list[camIdx], (1080, 1920))
                renderer_set.append(renderer)
            else:
                renderer_set.append(None)

        self.renderer_set = renderer_set

        ## load hand & object template mesh ##
        # self.load_hand_mesh()
        self.obj_mesh_data = self.load_obj_mesh()

        ## load annotation and images from filtered csv ##
        # self.anno_dict, self.rgb_dict, self.depth_dict = self.load_data(base_anno, base_source, seq, trialName, valid_cams)

        ### (extra) multi-view data from origin db ###
        origin_base_path = os.path.join(baseDir, 'origin', seq, trialName)
        self.anno_dict, self.rgb_dict, self.depth_dict, self.origin_dict = self.load_data_with_origin(base_anno, base_source, seq, trialName,
                                                                        valid_cams, origin_base_path)


        ## load deeplab model for segmentation##
        opts = deeplab_opts(int(self.obj_id), device)
        model, self.transform, self.decode_fn = deepSegLoadModel(opts)
        self.segModel = model.eval()

        ## set sample list with segmentation results ##
        # self.samples = self.set_sample()
        # annotation 3D pose에 접근해서, 해당되는 cam으로 projection한 뒤, 2D projected point로 bbox 추출
        self.samples = self.set_sample_origin()


    def __len__(self):
        return len(self.anno_dict)

    def __getitem__(self, queue):
        frame, camIdx = queue
        try:
            sample = self.samples[frame][camIDset[camIdx]]
        except:
            raise "Error at camIdx - frame {}".format(camIdx, frame)
        return sample

    def set_sample(self):
        samples = {}
        for camIdx, camID in enumerate(camIDset):
            if camID in self.valid_cams:
                samples[camID] = []

                for idx in range(self.get_len(camID)):
                    sample = {}

                    rgb = self.rgb_dict[camID][idx]
                    depth = self.depth_dict[camID][idx]

                    anno = self.anno_dict[camID][idx]
                    hand_2d = np.squeeze(np.asarray(anno['hand']['projected_2D_pose_per_cam']))
                    bbox, bbox_s = extractBbox(hand_2d)

                    rgb = rgb[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    depth = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

                    seg, vis_seg = deepSegPredict(self.segModel, self.transform, rgb, self.decode_fn, self.device)
                    # vis_seg = np.squeeze(np.asarray(vis_seg))
                    # hand_mask = np.asarray(vis_seg[:, :, 0] / 128 * 255, dtype=np.uint8)
                    # cv2.imshow("vis_seg", hand_mask)
                    # cv2.waitKey(0)

                    seg = np.asarray(seg)

                    # seg_hand = np.where(seg == 1, 1, 0)
                    seg_obj = np.where(seg == 2, 1, 0)

                    depth_obj = depth.copy()
                    depth_obj[seg_obj == 0] = 0
                    depth[seg == 0] = 0

                    # change depth image to m scale and background value as positive value
                    depth /= 1000.
                    depth_obj /= 1000.

                    depth_obj = np.where(seg != 2, 10, depth)
                    # depth_hand = np.where(seg != 1, 10, depth)

                    rgb = torch.FloatTensor(rgb).to(self.device)
                    depth_obj = torch.unsqueeze(torch.FloatTensor(depth_obj), 0).to(self.device)
                    seg_obj = torch.unsqueeze(torch.FloatTensor(seg_obj), 0).to(self.device)
                    # depth = torch.unsqueeze(torch.FloatTensor(depth), 0).to(self.device)

                    sample['rgb'], sample['depth_obj'], sample['seg_obj'] = rgb, depth_obj, seg_obj
                    sample['bb'] = [int(bb) for bb in bbox]

                    samples[camID].append(sample)

        return samples

    def set_sample_origin(self):
        samples = {}

        for key in self.anno_dict.keys():
            samples[key] = {}

            anno_cams = self.anno_dict[key]
            valid_cam_list = list(anno_cams.keys())
            anno = anno_cams[valid_cam_list[0]]

            for camIdx, camID in enumerate(camIDset):
                sample = {}

                rgb = self.origin_dict['rgb'][key][camID]
                depth = self.origin_dict['depth'][key][camID]

                mano_3D_joints = torch.FloatTensor(np.squeeze(np.asarray(anno['annotations'][0]['data']))).to(self.device)
                joints_cam = torch.unsqueeze(torch.Tensor(mano3DToCam3D(mano_3D_joints, self.Ms_list[camIdx])), axis=0)
                pred_kpts2d = projectPoints(joints_cam, self.Ks_list[camIdx]).cpu().detach().numpy()
                pred_kpts2d = np.squeeze(pred_kpts2d)

                bbox, bbox_s = extractBbox(pred_kpts2d)

                rgb = rgb[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                depth = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

                seg, vis_seg = deepSegPredict(self.segModel, self.transform, rgb, self.decode_fn, self.device)
                # vis_seg = np.squeeze(np.asarray(vis_seg))
                # hand_mask = np.asarray(vis_seg[:, :, 0] / 128 * 255, dtype=np.uint8)
                # cv2.imshow("vis_seg", hand_mask)
                # cv2.waitKey(0)

                seg = np.asarray(seg)
                # seg_hand = np.where(seg == 1, 1, 0)
                seg_obj = np.where(seg == 2, 1, 0)

                depth_obj = depth.copy()
                depth_obj[seg_obj == 0] = 0
                depth[seg == 0] = 0

                # change depth image to m scale and background value as positive value
                depth /= 1000.
                depth_obj /= 1000.

                depth_obj = np.where(seg != 2, 10, depth)
                # depth_hand = np.where(seg != 1, 10, depth)

                rgb = torch.FloatTensor(rgb).to(self.device)
                depth_obj = torch.unsqueeze(torch.FloatTensor(depth_obj), 0).to(self.device)
                seg_obj = torch.unsqueeze(torch.FloatTensor(seg_obj), 0).to(self.device)
                # depth = torch.unsqueeze(torch.FloatTensor(depth), 0).to(self.device)

                sample['rgb'], sample['depth_obj'], sample['seg_obj'] = rgb, depth_obj, seg_obj
                sample['bb'] = [int(bb) for bb in bbox]

                samples[key][camID] = sample

        return samples


    def load_hand_mesh(self):
        from manopth.manolayer import ManoLayer
        mano_path = os.path.join(os.getcwd(), 'modules', 'mano', 'models')
        self.mano_layer = ManoLayer(side='right', mano_root=mano_path, use_pca=False, flat_hand_mean=True,
                               center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)
        self.hand_faces_template = self.mano_layer.th_faces.repeat(1, 1, 1)

    def load_obj_mesh(self):
        target_mesh_class = str(self.obj_id) + '_' + str(OBJType(int(self.obj_id)).name)
        self.obj_mesh_name = target_mesh_class + '.obj'

        obj_mesh_path = os.path.join(self.baseDir, objModelDir, target_mesh_class, self.obj_mesh_name)
        obj_scale = CFG_OBJECT_SCALE_FIXED[int(self.obj_id) - 1]
        obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
        obj_verts_template = (obj_verts * float(obj_scale)).to(self.device)
        obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(self.device)

        # h = torch.ones((obj_verts_template.shape[0], 1), device=self.device)
        # self.obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

        obj_mesh_data = {}
        obj_mesh_data['verts'] = obj_verts_template
        obj_mesh_data['faces'] = obj_faces_template

        return obj_mesh_data


    def load_data(self, base_anno, base_source, seq, trialName, valid_cams):
        if self.csv_list is not None:
            df = self.csv_list.loc[self.csv_list['Sequence'] == seq]
            df = df.loc[df['Trial'] == trialName]
            filtered_list = np.asarray(df['Frame'])
        else:
            filtered_list = None

        anno_base_path = os.path.join(base_anno, seq, trialName, 'annotation')
        rgb_base_path = os.path.join(base_source, seq, trialName, 'rgb')
        depth_base_path = os.path.join(base_source, seq, trialName, 'depth')

        anno_dict = {}
        rgb_dict = {}
        depth_dict = {}
        for camIdx, camID in enumerate(camIDset):
            anno_dict[camID] = []
            rgb_dict[camID] = []
            depth_dict[camID] = []

            if camID in valid_cams:
                anno_list = os.listdir(os.path.join(anno_base_path, camID))
                for anno in anno_list:
                    if filtered_list is not None and anno[:-5] in filtered_list:
                        anno_path = os.path.join(anno_base_path, camID, anno)
                        with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                            anno_data = json.load(file)
                            anno_dict[camID].append(anno_data)

                        rgb_path = os.path.join(rgb_base_path, camID, anno[:-5] + '.jpg')
                        rgb_data = np.asarray(cv2.imread(rgb_path))
                        rgb_dict[camID].append(rgb_data)

                        depth_path = os.path.join(depth_base_path, camID, anno[:-5] + '.png')
                        depth_data = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype(float)
                        depth_dict[camID].append(depth_data)

        return anno_dict, rgb_dict, depth_dict


    def get_obj_pose(self, frame):
        anno_dict = self.anno_dict[frame]
        key = list(anno_dict.keys())[0]
        obj_mat = np.squeeze(np.asarray(anno_dict[key]['Mesh'][0]['object_mat']))
        return obj_mat


    def load_data_with_origin(self, base_anno, base_source, seq, trialName, valid_cams, origin_base_path):
        if self.csv_list is not None:
            df = self.csv_list.loc[self.csv_list['Sequence'] == seq]
            df = df.loc[df['Trial'] == trialName]
            filtered_list = np.asarray(df['Frame'])
        else:
            filtered_list = None

        anno_base_path = os.path.join(base_anno, seq, trialName, 'annotation')
        rgb_base_path = os.path.join(base_source, seq, trialName, 'rgb')
        depth_base_path = os.path.join(base_source, seq, trialName, 'depth')

        anno_dict = {}
        rgb_dict = {}
        depth_dict = {}

        origin_dict = {}
        origin_dict['rgb'] = {}
        origin_dict['depth'] = {}

        for camIdx, camID in enumerate(camIDset):
            if camID in valid_cams:
                anno_list = os.listdir(os.path.join(anno_base_path, camID))
                for anno in anno_list:
                    file_name = anno[:-5]
                    if filtered_list is not None and file_name in filtered_list:
                        file_idx = file_name.split('_')[-1]
                        if file_idx not in anno_dict:
                            anno_dict[file_idx] = {}

                        anno_path = os.path.join(anno_base_path, camID, anno)
                        with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                            anno_data = json.load(file)
                            anno_dict[file_idx][camID] = anno_data

                        if file_idx not in origin_dict['rgb']:
                            origin_dict['rgb'][file_idx] = {}
                            origin_dict['depth'][file_idx] = {}

                            for camID_ori in camIDset:
                                origin_path_rgb = os.path.join(origin_base_path, 'rgb', camID_ori, camID_ori+'_'+file_idx+'.jpg')
                                origin_path_depth = os.path.join(origin_base_path, 'depth', camID_ori, camID_ori+'_'+file_idx + '.png')

                                origin_rgb = np.asarray(cv2.imread(origin_path_rgb))
                                origin_depth = np.asarray(cv2.imread(origin_path_depth, cv2.IMREAD_UNCHANGED)).astype(float)

                                origin_dict['rgb'][file_idx][camID_ori] = origin_rgb
                                origin_dict['depth'][file_idx][camID_ori] = origin_depth

        return anno_dict, rgb_dict, depth_dict, origin_dict



def main():
    from natsort import natsorted
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### N5 path / N1 path ###
    baseDir = os.path.join('dataset/NIA_db')
    # baseDir = os.path.join('/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets')

    base_source = os.path.join(baseDir, '1_Source_data')
    base_anno = os.path.join(baseDir, '2_Labeling_data')

    seq_list = natsorted(os.listdir(base_anno))
    print("total sequence # : ", len(seq_list))

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


if __name__ == '__main__':
    main()