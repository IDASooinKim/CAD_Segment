# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_iou(
    ground_thruth: torch.tensor,
    prediction: torch.tensor
):
    
    r"""
        [NOTICE]
            Tensor shape should be (B x C x W x H)
            B : Batch size
            C : Channels
            W : Width of tensor
            H : Height of tensor
    """
    ground_thruth = torch.squeeze(ground_thruth)
    ground_thruth = F.softmax(ground_thruth, dim=0).detach().cpu().numpy()
    ground_thruth = np.where(
        ground_thruth == np.max(ground_thruth, axis=0), 1, 0
    )

    prediction = torch.squeeze(prediction)
    prediction = F.softmax(prediction, dim=0).detach().cpu().numpy()
    prediction = np.where(
        prediction == np.max(prediction, axis=0), 1, 0
    )

    average_ioU = float(0)

    for class_idx in range(ground_thruth.shape[0]):
        
        class_overlap = np.where(
            ground_thruth[class_idx, :, :]+prediction[class_idx, :, :] == 2, 1, 0
        )
        
        class_intersection = np.where(
            ground_thruth[class_idx, :, :]+prediction[class_idx, :, :] >= 1, 1, 0
        )
        
        average_ioU += float(
            np.sum(class_overlap)/np.sum(class_intersection)
        )

    return average_ioU
