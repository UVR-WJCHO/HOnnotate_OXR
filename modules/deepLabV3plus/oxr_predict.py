import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(), 'modules'))
from torch.utils.data import dataset
from tqdm import tqdm
import deepLabV3plus.network as network
import deepLabV3plus.utils as utils
import os
import random
import argparse
import numpy as np
import pickle as pkl
import time
import cv2
from torch.utils import data
from deepLabV3plus.datasets import VOCSegmentation, Cityscapes, cityscapes, OXRSegmentation
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def predict(rgb, opts):
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = network.modeling.__dict__[opts.model](21, output_stride=opts.output_stride)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model.classifier.classifier[3] = nn.Conv2d(256,3,1)

    model.load_state_dict(opts.checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    decode_fn = OXRSegmentation.decode_target
    with torch.no_grad():
        model = model.eval()
        #TODO check image format
        b = rgb[:,:,0]
        g = rgb[:,:,1]
        r = rgb[:,:,2]
        rgb = np.stack([r,g,b], axis=2)
        img = transform(rgb).unsqueeze(0)
        img = img.to(device)
        pred = model(img)
        mask = pred.max(1)[1].cpu().numpy()[0]
        mask_img = Image.fromarray(mask.astype('uint16'))
        colorized_preds = decode_fn(mask).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds)

    return mask_img, colorized_preds



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='oxr',
                        choices=['voc', 'cityscapes'], help='Name of training set')
    parser.add_argument("--cam", type=str, default='mas', choices=['mas', 'sub1', 'sub2', 'sub3'])
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == "oxr":
        opts.num_calsses = 2
        decode_fn = OXRSegmentation.decode_target

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                print(len(files))
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](21, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model.classifier.classifier[3] = nn.Conv2d(256, 3, 1)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        raise "NotImplementedError"
        model = nn.DataParallel(model)
        model.to(device)

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    gaps = []
    start = time.time()
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            pred = model(img) #[1, 3, 480, 640]
            # raw = pred[0]
            # pkl.dump(raw.cpu().numpy(), open(os.path.join(opts.save_val_results_to, "raw_seg", opts.cam, img_name+'.pkl'), 'wb'))
            # np.save(os.path.join(opts.save_val_results_to, opts.cam , "raw_seg", img_name+'.npz'), raw.cpu().numpy())
            # raw_img = Image.fromarray(raw.cpu().numpy().astype('uint8').transpose(1,2,0))
            # raw_img.save(os.path.join(opts.save_val_results_to, "raw_seg", opts.cam, img_name+'.png'))
            pred = pred.max(1)[1].cpu().numpy()[0] # HW [480, 640]
            mask_img = Image.fromarray(pred.astype('uint16'))
            mask_img.save(os.path.join(opts.save_val_results_to, opts.cam,"mask_seg", img_name+'.png'))
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            end = time.time()
            gaps.append(end-start)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, opts.cam, "vis_seg", img_name+'.jpg'))
    print("Average time: %f" % np.mean(gaps))
        
if __name__ == '__main__':
    main()