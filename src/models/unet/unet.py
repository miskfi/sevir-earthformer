"""Code adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import pytorch_lightning as pl
import torch.nn as nn

from .unet_blocks import DecoderBlock, DoubleConvBlock, EncoderBlock


class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.in_conv = DoubleConvBlock(in_channels, 32)
        self.down1 = EncoderBlock(32, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = EncoderBlock(256, 512 // factor)

        self.up1 = DecoderBlock(512, 256 // factor, bilinear)
        self.up2 = DecoderBlock(256, 128 // factor, bilinear)
        self.up3 = DecoderBlock(128, 64 // factor, bilinear)
        self.up4 = DecoderBlock(64, 32, bilinear)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
