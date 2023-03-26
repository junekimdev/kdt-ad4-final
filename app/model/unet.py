import torch
import torch.nn as nn
from app.model.conv import RegularConv, SeparableConv
from app.config import Config
config = Config()


class EncoderBlockReg(nn.Module):
    """A Block for Encoder with Regular Convolution"""

    # config injection
    def __init__(self, in_channels, out_channels,
                 kernel_size=config.pool_kernel, stride=config.pool_stride):
        super().__init__()

        self.cov1 = RegularConv(in_channels, out_channels)
        self.cov2 = RegularConv(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        p = self.pool(x)
        return x, p


class EncoderBlockSep(nn.Module):
    """A Block for Encoder with Separable Convolution"""

    # config injection
    def __init__(self, in_channels, out_channels,
                 kernel_size=config.pool_kernel, stride=config.pool_stride):
        super().__init__()

        self.cov1 = SeparableConv(in_channels, out_channels)
        self.cov2 = SeparableConv(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        p = self.pool(x)
        return x, p


class DecoderBlockReg(nn.Module):
    """A Block for Decoder with Regular Convolution"""

    # config injection
    def __init__(self, in_channels, out_channels,
                 kernel_size=config.upsample_kernel, stride=config.upsample_stride):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        # in_channels == out_channels*2
        self.cov1 = SeparableConv(in_channels, out_channels)
        self.cov2 = RegularConv(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.concat([x, skip], dim=1)
        x = self.cov1(x)
        x = self.cov2(x)
        return x


class DecoderBlockSep(nn.Module):
    """A Block for Decoder with Separable Convolution"""

    # config injection
    def __init__(self, in_channels, out_channels,
                 kernel_size=config.upsample_kernel, stride=config.upsample_stride):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride)
        # in_channels == out_channels*2
        self.cov1 = SeparableConv(in_channels, out_channels)
        self.cov2 = SeparableConv(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.concat([x, skip], dim=1)
        x = self.cov1(x)
        x = self.cov2(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, init_feature=config.init_feature):
        super().__init__()
        self.ec1 = EncoderBlockReg(in_channels, init_feature)
        self.ec2 = EncoderBlockSep(init_feature, init_feature*2)
        self.ec3 = EncoderBlockSep(init_feature*2, init_feature*4)
        self.ec4 = EncoderBlockSep(init_feature*4, init_feature*8)

        self.br = EncoderBlockSep(init_feature*8, init_feature*16)

        self.dc4 = DecoderBlockSep(init_feature*16, init_feature*8)
        self.dc3 = DecoderBlockSep(init_feature*8, init_feature*4)
        self.dc2 = DecoderBlockSep(init_feature*4, init_feature*2)
        self.dc1 = DecoderBlockReg(init_feature*2, init_feature)

        self.conv = nn.Conv2d(init_feature, out_channels, kernel_size=1)

    def forward(self, x):
        skip1, x = self.ec1(x)
        skip2, x = self.ec2(x)
        skip3, x = self.ec3(x)
        skip4, x = self.ec4(x)
        x, _ = self.br(x)
        x = self.dc4(x, skip4)
        x = self.dc3(x, skip3)
        x = self.dc2(x, skip2)
        x = self.dc1(x, skip1)
        x = self.conv(x)
        return x
