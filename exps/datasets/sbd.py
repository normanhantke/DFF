###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from .base_sbd import BaseDataset

class SBDEdgeDetection(BaseDataset):
    NUM_CLASS = 20
    def __init__(self, root='../data/sbd-preprocess/data_proc', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(SBDEdgeDetection, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        assert os.path.exists(root), "Please download the dataset and place it under: %s"%root

        self.images, self.masks = _get_sbd_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        if self.mode == 'testval':
            img_size = torch.from_numpy(np.array(img.size))
            img = self._testval_sync_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index]), img_size
        
        mask = Image.open(self.masks[index])

        if self.mode == 'vis':
            img_size = torch.from_numpy(np.array(img.size))
            img, mask = self._vis_sync_transform(img, mask)
            if self.transform is not None:
                img = self.transform(img)
            return img, mask, os.path.basename(self.images[index]), img_size
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            assert self.mode == 'val'
            img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask


    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_sbd_pairs(folder, split='train'):
    def get_path_pairs(folder,split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in lines:
                ll_str = re.split(' ', line)
                imgpath = folder + ll_str[0].rstrip()
                maskpath = folder + ll_str[1].rstrip()
                if os.path.isfile(imgpath):
                    img_paths.append(imgpath)
                else:
                    print('cannot find the image:', imgpath)
                if os.path.isfile(maskpath):
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths
    if split == 'train':
        split_f = os.path.join(folder, 'trainvalaug_inst_orig.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'test_inst_orig.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        split_f = os.path.join(folder, 'vis.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)

    return img_paths, mask_paths
