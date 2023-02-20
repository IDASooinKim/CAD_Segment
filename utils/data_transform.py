# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import random
import torch

from typing import Sequence

import torchvision.transforms.functional as TF


class SelectRotation:
    
    def __init__(self,
        angles: Sequence[int]
    ):
        self.angles=angles

    def __call__(self, 
        x: torch.tensor
    ):
        angle = random.choice(self.angles)

        return TF.rotate(x, angle)
