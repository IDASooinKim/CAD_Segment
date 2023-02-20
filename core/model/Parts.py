# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    
    r"""
        [Discription]
            Build convolutional block. The block consists of conv-batch_norm-activation.

        [Attribute]
            in_channels: number of kernels input channels will have
            out_channels: number of kernels output channels will have
            mid_channels: number of kernels middle channels will have
    """
    
    def __init__(self,
        in_channels: int=3,
        out_channels: int=3,
        mid_channels: int=None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=mid_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, 
        x: torch.Tensor
    ):
        return self.double_conv(x)


class Down(nn.Module):

    r"""
        [Discription]
            Build Down-sampling Block. the Block consists of maxpool-double_conv.

        [Attribute]
            in_channels: number of kernels input channels will have 
            out_channels: number of kernels output channels will have
    """

    def __init__(self,
        in_channels: int=3,
        out_channels: int=3,
    ):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        )

    def forward(self,
        x: torch.Tensor
    ):
        return self.maxpool_conv(x)


class Up(nn.Module):

    r"""
        [Discription]
            Build Up-sampling Block. the Block consists of upsamle-double_conv.

        [Attribute]
            in_channels: number of kernels input channels will have 
            out_channels: number of kernels output channels will have
            bilinear: boolean value for select linear interpolation
    """

    def __init__(self,
        in_channels: int=3,
        out_channels: int=3,
        bilinear: bool=True
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=in_channels//2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels//2,
                kernel_size=2,
                stride=2
            )
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
    
    def forward(self, 
        x1: torch.Tensor,
        x2: torch.Tensor
    ):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):

    r"""
        [Discription]
            Build out convolutional layer.

        [Attribute]
            in_channels: number of kernels input channels will have 
            out_channels: number of kernels output channels will have
    """

    def __init__(self,
        in_channels: int=3,
        out_channels: int=3 
    ):
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, 
        x: torch.Tensor
    ):
        return self.conv(x)
