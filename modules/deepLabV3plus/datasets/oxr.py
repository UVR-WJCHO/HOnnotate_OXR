import os
import sys
import torch.utils.data as data
import numpy as np
from PIL import Image

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# root = /home/workplace/HOnnotate_OXR/dataset/oxr_seg_data
# -- mask
# -- rgb
class OXRSegmentation(data.Dataset):
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 is_aug=False):
        self.root = root
        self.transform = transform
        self.image_set = image_set
        self.is_aug = is_aug
        self.imgpath = os.path.join(self.root, 'rgb')
        self.maskpath  = os.path.join(self.root, 'mask')

        self.images = [os.path.join(self.imgpath, img) for img in os.listdir(os.path.join(self.imgpath)) if img.endswith('.png')]
        self.masks = [os.path.join(self.maskpath, img) for img in os.listdir(os.path.join(self.maskpath)) if img.endswith('.png')]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        target = np.array(target)
        target = target / 125.0
        target = target.astype('uint8')
        # tar1 = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
        # tar2 = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
        # tar1 = np.where(target == 1, 1, 0)
        # tar2 = np.where(target == 2, 1, 0)
        # tar = np.stack((tar1, tar2), axis=0)

        return img, target

    def __len__(self):
        return len(self.images)
    
    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]
    
    
