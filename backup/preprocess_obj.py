import os
import sys

sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models'))
sys.path.insert(0,os.path.join(os.getcwd(), 'HOnnotate_refine/models/slim'))

from absl import flags
from absl import app
import numpy as np
# others
import cv2
from PIL import Image
import time
import json
import pickle
import copy
import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

### FLAGS ###
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '230905', 'target db Name')   ## name ,default, help

camIDset = ['mas', 'sub1', 'sub2', 'sub3']
FLAGS(sys.argv)

### Config ###
baseDir = os.path.join(os.getcwd(), 'dataset')
bgmModelPath = os.path.join(os.getcwd(), 'modules', 'pretrained', 'model.pth')
"""
Background matting을 위해 pretrained model 다운
!pip install gdown -q
!gdown https://drive.google.com/uc?id=1-t9SO--H4WmP7wUl1tVNNeDkq47hjbv4 -O model.pth -q
"""
camResultDir = os.path.join(baseDir, FLAGS.db)
image_cols, image_rows = 1080, 1920

class loadDataset(Dataset):
    def __init__(self, db, seq, trial, camID):
        self.seq = seq # 230612_S01_obj_01_grasp_01
        self.db = db
        assert len(seq.split('_')) == 6, 'invalid sequence name, ex. 230612_S01_obj_01_grasp_01'

        self.subject_id = seq.split('_')[1][1:]
        self.obj_id = seq.split('_')[3]
        self.grasp_id = seq.split('_')[5]
        self.trial = trial
        self.trial_num = trial.split('_')[1]

        self.dbDir = os.path.join(baseDir, db, seq, trial)
        self.rgbDir = os.path.join(self.dbDir, 'rgb_crop')
        self.metaDir = os.path.join(self.dbDir, 'meta')
        self.bgDir = os.path.join(baseDir, db+'_bg')
        self.segDir = os.path.join(self.dbDir, 'segmentation')
        self.camID = camID
        self.load_dataset()
    
    def load_dataset(self):
        self.filelist = os.listdir(os.path.join(self.rgbDir, self.camID))

    def get_rgb(self, idx):
        filename = self.filelist[idx]
        rgb = cv2.imread(os.path.join(self.rgbDir, self.camID, filename))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = Image.fromarray(rgb)
        rgb = to_tensor(rgb)
        return rgb
    
    def get_bg(self, idx):
        filename = self.filelist[idx]
        bgr = cv2.imread(os.path.join(self.bgDir, 'rgb', self.camID ,'%s_4.jpg'%self.camID))
        meta_path = os.path.join(self.metaDir, self.camID, '%s.pkl' % os.path.splitext(filename)[0])
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        bbox = meta['bb']
        bgr = bgr[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = Image.fromarray(bgr)
        bgr = to_tensor(bgr)
        return bgr
    
    def get_seg(self, idx):
        filename = self.filelist[idx]
        seg = cv2.imread(os.path.join(self.segDir, self.camID, 'raw_seg_results', filename))
        # seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        seg = Image.fromarray(seg)
        seg = to_tensor(seg)
        return seg

    def get_sample(self, idx):
        sample = {}
        filename = self.filelist[idx]
        sample['rgb'] = self.get_rgb(idx)
        sample['bg'] = self.get_bg(idx)
        sample['seg'] = self.get_seg(idx)
        sample['file_name'] = os.path.splitext(filename)[0]
        return sample
    
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return sample
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.rgbDir, self.camID)))
    
class BackgroundMatting():
    def __init__(self, db, seq, trial, camID):
        self.db = db
        self.seq = seq
        self.trial = trial
        self.camID = camID
        self.dataset = loadDataset(db, seq, trial, camID)
        self.dataLoader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
        self.resultDir = os.path.join(baseDir, db, seq, trial, 'segmentation_fg', camID)
        self.objMaskDir = os.path.join(baseDir, db, seq, trial, 'segmentation_obj', camID)

        os.makedirs(self.resultDir, exist_ok=True)
        os.makedirs(self.objMaskDir, exist_ok=True)

    def run(self):
        for sample in self.dataLoader:
            model = torch.jit.load(bgmModelPath).cuda().eval()
            rgb = sample['rgb'].cuda()
            bgr = sample['bg'].cuda()
            if rgb.size(2) <= 2048 and rgb.size(3) <= 2048:
                model.backbone_scale = 1/4
                model.refine_sample_pixels = 80_000
            else:
                model.backbone_scale = 1/8
                model.refine_sample_pixels = 320_000
            pha, fgr = model(rgb, bgr)[:2]
            for idx, file_name in enumerate(sample['file_name']):
                pha_ = pha[idx]
                fgr_ = fgr[idx]
                mask = pha_.permute(1, 2, 0).cpu().numpy()*255
                com = pha_ * fgr_ + (1 - pha_) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
                com = com.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
                com = cv2.cvtColor(com.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.resultDir, '%s.jpg' % file_name), mask)
                cv2.imwrite(os.path.join(self.resultDir, '%s_masked.png' % file_name), com)

                seg = sample['seg'][idx]*255
                seg = seg.permute(1, 2, 0).cpu().numpy()
                seg = np.squeeze(seg[:, :, 0])
                mask = np.squeeze(mask)
                mask[mask < 200] = 0
                mask[seg > 10] = 0
                mask = np.where(mask > 1, 255, 0)
                mask = mask.astype(np.uint8)
                cv2.imwrite(os.path.join(self.objMaskDir, '%s.jpg' % file_name), mask)


if __name__ == '__main__':
    t1 = time.time()
    for seq in os.listdir(camResultDir):
        for trial in os.listdir(os.path.join(camResultDir, seq)):
            for camID in camIDset:
                print("run : ", seq, trial, camID)
                bgm = BackgroundMatting(FLAGS.db, seq, trial, camID)
                bgm.run()

    proc_time = round((time.time() - t1) / 60., 2)
    print("total process time : %s min" % (str(proc_time)))