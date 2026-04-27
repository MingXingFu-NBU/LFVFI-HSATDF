import os
import cv2
import ast
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import torchvision.transforms as transforms
from PIL import Image
import imageio
from einops import rearrange
from functions.utils import ColorAugmentation


##################################### training dataset #####################################
class TrainSetLoader(Dataset):
    def __init__(self, args):
        self.path = args.trainset_path
        self.filenames = [os.path.join(args.trainset_path, f) for f in os.listdir(args.trainset_path) if
                          not f.startswith('.')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        folder_path = self.filenames[idx]
        with h5py.File(folder_path, 'r') as hf:
            img0 = np.array(hf.get('frame_0')).astype(np.float32) / 255.   # [u,v,c,h,w]
            img1 = np.array(hf.get('frame_1')).astype(np.float32) / 255.
            gt = np.array(hf.get('frame_t')).astype(np.float32) / 255.

        # 4D Augmentation
        # flip
        if np.random.rand(1) > 0.5:
            img0 = np.flip(np.flip(img0, 0), 3)
            img1 = np.flip(np.flip(img1, 0), 3)
            gt = np.flip(np.flip(gt, 0), 3)
        if np.random.rand(1) > 0.5:
            img0 = np.flip(np.flip(img0, 1), 4)
            img1 = np.flip(np.flip(img1, 1), 4)
            gt = np.flip(np.flip(gt, 1), 4)
        # rotate
        r_ang = np.random.randint(1, 5)
        img0 = np.rot90(img0, r_ang, (3, 4))
        img0 = np.rot90(img0, r_ang, (0, 1))
        img1 = np.rot90(img1, r_ang, (3, 4))
        img1 = np.rot90(img1, r_ang, (0, 1))
        gt = np.rot90(gt, r_ang, (3, 4))
        gt = np.rot90(gt, r_ang, (0, 1))
        # color
        c_ang = np.random.randint(1, 7)
        img0 = ColorAugmentation(img0, c_ang)
        img1 = ColorAugmentation(img1, c_ang)
        gt = ColorAugmentation(gt, c_ang)

        # Get input and label   [c,ah,aw,h,w]
        img0 = np.transpose(img0, [2, 0, 1, 3, 4])
        img1 = np.transpose(img1, [2, 0, 1, 3, 4])
        gt = np.transpose(gt, [2, 0, 1, 3, 4])

        # Convert to tensor
        img0 = torch.from_numpy(img0.copy())
        img1 = torch.from_numpy(img1.copy())
        gt = torch.from_numpy(gt.copy())

        return img0, img1, gt


##################################### test dataset #####################################
class TestSetLoader(Dataset):
    def __init__(self, cfg):
        self.path = cfg.testset_path
        self.file_folder = [os.path.join( cfg.testset_path, f) for f in os.listdir( cfg.testset_path) if not f.startswith('.')]
        self.filenames0 = [os.path.join(self.file_folder[0], f) for f in os.listdir(self.file_folder[0]) if
                          not f.startswith('.')]
        self.filenames1 = [os.path.join(self.file_folder[1], f) for f in os.listdir(self.file_folder[1]) if
                          not f.startswith('.')]
        self.filenames0_folder = [os.path.join(self.filenames0[0], f) for f in os.listdir(self.filenames0[0]) if
                          not f.startswith('.')]
        self.filenames1_folder = [os.path.join(self.filenames1[0], f) for f in os.listdir(self.filenames1[0]) if
                          not f.startswith('.')]
        self.filenames=[*self.filenames0_folder+self.filenames1_folder]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        folder_path = self.filenames[idx]
        img0_path = os.path.join(folder_path,'01.png')
        gt_path = os.path.join(folder_path, '02.png')
        img1_path = os.path.join(folder_path, '03.png')
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(folder_path)))

        img0 = np.array(Image.open(img0_path))
        gt = np.array(Image.open(gt_path))
        img1 = np.array(Image.open(img1_path))

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1) / 255.
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1) / 255.
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1) / 255.

        img0 = extract_subaperture_views(img0, angular_res=5)    # [u,v,c,h,w]
        img1 = extract_subaperture_views(img1, angular_res=5)
        gt = extract_subaperture_views(gt, angular_res=5)

        img0 = img0.permute(2, 0, 1, 3, 4).contiguous()
        img1 = img1.permute(2, 0, 1, 3, 4).contiguous()
        gt = gt.permute(2, 0, 1, 3, 4).contiguous()

        if dataset_name == 'dataset3_3frame':
            img0 = rearrange(img0, 'c an1 an2 h w -> c (an1 an2) h w', c=3, an1=5, an2=5, h=1080, w=1920)
            img1 = rearrange(img1, 'c an1 an2 h w -> c (an1 an2) h w', c=3, an1=5, an2=5, h=1080, w=1920)
            gt = rearrange(gt, 'c an1 an2 h w -> c (an1 an2) h w', c=3, an1=5, an2=5, h=1080, w=1920)
            img0 = F.interpolate(
                input=img0, size=(360, 640),
                mode="bicubic", align_corners=False)
            img1 = F.interpolate(
                input=img1, size=(360, 640),
                mode="bicubic", align_corners=False)
            gt = F.interpolate(
                input=gt, size=(360, 640),
                mode="bicubic", align_corners=False)
            img0 = rearrange(img0, 'c (an1 an2) h w ->c an1 an2 h w', c=3, an1=5, an2=5, h=360, w=640)
            img1 = rearrange(img1, 'c (an1 an2) h w ->c an1 an2 h w', c=3, an1=5, an2=5, h=360, w=640)
            gt = rearrange(gt, 'c (an1 an2) h w ->c an1 an2 h w', c=3, an1=5, an2=5, h=360, w=640)
        return img0, img1, gt


def extract_subaperture_views(light_field_img: torch.Tensor, angular_res: int = 5):
    C, H, W = light_field_img.shape
    h = H // angular_res
    w = W // angular_res

    views = torch.zeros((angular_res, angular_res, C, h, w), dtype=light_field_img.dtype)

    for u in range(angular_res):
        for v in range(angular_res):
            views[u, v] = light_field_img[:, u::angular_res, v::angular_res]

    return views