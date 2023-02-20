# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

from .Parts import *
from torchsummary import summary


class UNet(nn.Module):

    r"""
        [Discription]
            Build Unet.

        [Attribute]
            n_channels: number of kernels input channels will have
            n_class: number of kernels output channels will have
            bilinear: boolean value for select linear interpolation
    """

    def __init__(self,
        n_channels: int=3,
        n_classes: int=10,
        bilinear: bool=False
    ):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


if __name__ == '__main__':

    model = UNet(
        n_channels=3, 
        n_classes=5, 
        bilinear=False
    ).to("cpu")

    summary(model, 
        input_size=(3,256,256),
        batch_size=1,
        device='cpu'
    )
