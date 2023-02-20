# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import random
import warnings
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from core.model import UNet
from utils.data_loader import ImageDataset
from utils.infer_model import inference
from interface.argparse import get_args

def main():

    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            f'\n[Info] You have chosen specific seed : {args.seed}'
        )
    
    if args.save_infer_result is None:
        warnings.warn(
            f'[Warning] You have to choose specific directory for saving results'
        )
        sys.exit(1)
    
    main_worker(
        args.save_infer_result,
        args
    )

def main_worker(
    saving_path: str,
    args: object
):

    #Preparing and loading data
    test_dataset = ImageDataset(
        data_path=args.img_path,
        transform=None,
        scaling=True,
        mode='test'
    )

    #Load data with cpu
    test_data = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=None,
        shuffle=False,
        num_workers=args.num_workers
    )

    #Load model and set it into device
    machine = UNet(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        bilinear=False
    )

    machine.load_state_dict(
        torch.load(
            f=args.save_model_path + f'/{args.saved_model_name}',
            map_location=None
        ),
        strict=False
    )

    criterion = nn.MSELoss()

    inference(
        model=machine,
        data_loader=test_data,
        loss_fn=criterion,
        epoch=1,
        device=None,
        args=args
    )

if __name__ == '__main__':

    main()
