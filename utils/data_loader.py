# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import numpy as np
import random
import h5py
import sys

from typing import Sequence
from PIL import Image
from glob import glob

import torch
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self,
        data_path: str,
        transform: object=None,
        scaling: bool=True,
        mode: str='train'
    ):
        
        self.data_path = data_path
        self.transform = transform
        self.scaling = scaling
        self.mode = mode

        if self.mode == 'inference':
            self.infer_list = glob(self.data_path)

    def __len__(self):

        if self.mode == 'train':
            with h5py.File(self.data_path, 'r') as dataset:
                length = len(dataset['train_images'])
            dataset.close()
            return length
        
        elif self.mode == 'test':
            with h5py.File(self.data_path, 'r') as dataset:
                length = len(dataset['test_images'])
            dataset.close()
            return length
        
        elif self.mode == 'inference':
            return len(self.data_path)
        
        else:
            raise Exception(
                f'[ERROR] {self.mode} is unknown method. you can choose train or inference'
                )
            sys.exit(1)
            
    def __getitem__(self, 
                    index: int
                    ):
        if self.mode == 'train':
            with h5py.File(self.data_path, 'r') as dataset:
                image = torch.tensor(
                    dataset['train_images'][index], 
                    dtype=torch.float
                )
                label = torch.tensor(
                    dataset['train_labels'][index], 
                    dtype=torch.float
                )
                
        elif self.mode == 'test':
            with h5py.File(self.data_path, 'r') as dataset:
                image = torch.tensor(
                    dataset['test_images'][index], 
                    dtype=torch.float
                )
                label = torch.tensor(
                    dataset['test_labels'][index], 
                    dtype=torch.float
                )
        
        elif self.mode =='inference':
            image = Image.open(self.infer_list[index])
            label = self.infer_list[index][-1]

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        if self.scaling:
            image = image/255.
            label = label/255.
    
        image = image[np.newaxis, :, :]
    
        return image, label
