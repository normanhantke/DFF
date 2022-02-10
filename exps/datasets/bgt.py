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
from .base_bgt import BaseDataset

class BGTEdgeDetection(BaseDataset):
    NUM_CLASS = 3
    def __init__(self, root='../data', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(BGTEdgeDetection, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        root = os.path.join(root, 'bgt-dataset')
        assert os.path.exists(root), "Please download the dataset and place it under: %s"%root

        self.images, self.masks = _get_bgt_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        if self.mode == 'testval':
            img_size = torch.from_numpy(np.array(img.size))
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index]), img_size
        
        mask = Image.open(self.masks[index])

        if self.mode == 'vis':
            img_size = torch.from_numpy(np.array(img.size))
            if self.transform is not None:
                img = self.transform(img)
            mask = self._mask_transform(mask)
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

    def _mask_transform(self, mask):
        mask = torch.from_numpy(np.array(mask)).float()
        mask = mask.permute(2, 0, 1) # channel first

        return mask

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

def _get_bgt_pairs(folder, split='train'):

    if not os.path.exists(folder):
        print("Could not find the dataset directory!")

    if split == 'train':
        images_dir = os.path.join( os.path.join(folder, "train"), "images" )
        labels_dir = os.path.join( os.path.join(folder, "train"), "edges" )
    elif split == 'val':
        images_dir = os.path.join( os.path.join(folder, "val"), "images" )
        labels_dir = os.path.join( os.path.join(folder, "val"), "edges" )
    elif split == 'test':
        images_dir = os.path.join( os.path.join(folder, "test"), "images" )
        labels_dir = os.path.join( os.path.join(folder, "test"), "edges" )
    else:
        images_dir = os.path.join( os.path.join(folder, "vis"), "images" )
        labels_dir = os.path.join( os.path.join(folder, "vis"), "edges" )
        
    if not os.path.exists(images_dir):
        print("Could not find the images directory!")
    if not os.path.exists(labels_dir):
        print("Could not find the labels directory!")    
        
    img_paths = [ os.path.join( images_dir, f ) for f in os.listdir( images_dir ) if os.path.isfile( os.path.join(images_dir, f) ) ]
    img_paths.sort()
    mask_paths = [ os.path.join( labels_dir, f ) for f in os.listdir( labels_dir ) if os.path.isfile( os.path.join(labels_dir, f) ) ]
    mask_paths.sort()

    return img_paths, mask_paths
