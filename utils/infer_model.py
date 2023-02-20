# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 

import math
import sys
import numpy as np

from typing import Iterable, Optional
from time import time
from pkbar import Kbar


# color_palette = dict({
#     '0':[255, 0, 0],
#     '1':[0, 255, 0],
#     '2':[0, 0, 255],
#     '3':[0,255,255],
#     '4':[255, 0, 255],
#     '5':[255, 255, 0],
#     '6':[0, 0, 0],
#     '7':[125, 125, 125],
#     '8':[255, 255, 255],
# })

color_palette = list([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [0,255,255],
    [255, 0, 255],
    [255, 255, 0],
    [0, 0, 0],
    [125, 125, 125],
    [255, 255, 255],
])

def inference(
    model: nn.Module,
    data_loader: Iterable,
    loss_fn: nn.Module,
    epoch: int,
    device:object,
    args:object
):
    model.eval()

    for batch_idx, data in enumerate(data_loader):

        img, label = data
        img = img.to(
            device=device,
            non_blocking=True
            )
        label = label.to(
            device=device,
            non_blocking=True
            )

        # forward the function and extract the pred labels with automatic mixed-precision
        with torch.cuda.amp.autocast():
            pred_label = model(img)

            # calculation the loss and backpropagate for update weights.
            loss = loss_fn(
                pred_label,
                label
                )

        convert2image(
            gt_tensor=label,
            pred_tensor=pred_label,
            iteration=batch_idx,
            args=args
            )
 
    if args.rank == 0: 
        print(f'[INFO] Inference Loss is : {loss.item()}')

        #TODO: have to build metrics like, IoU 
    torch.cuda.synchronize()


def convert2image(
    gt_tensor: torch.tensor,
    pred_tensor: torch.tensor,
    iteration: int,
    args: object
):
    gt_label = gt_tensor.squeeze(0).detach().cpu().numpy()
    pred_label = pred_tensor.squeeze(0)

    pred_label = nn.Softmax(dim=0)(
        pred_label
        ).detach().cpu().numpy()

    pred_label_max = np.where(
        pred_label == np.max(pred_label, axis=0), 255, 0
        )

    for i in range(7):
        cv2.imwrite(f"./results/{iteration}_{i}.png", pred_label_max[i])
        cv2.imwrite(f"./results/{iteration}_{i}_gt.png", gt_label[i]*255)
    
    