import torch.nn as nn
from app.config import Config
config = Config()


class RegularConv(nn.Module):
    # config injection
    def __init__(self, in_channels, out_channels,
                 kernel_size=config.conv_kernel, stride=config.conv_stride,
                 padding=config.conv_padding, bias=config.conv_bias,
                 dropout_p=config.conv_dropout_p):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn = nn.BatchNorm2d(out_channels)

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn(x)
        return x


class SeparableConv(nn.Module):
    """
    @see: Xception: Deep Learning with Depthwise Separable Convolutions(https://arxiv.org/abs/1610.02357)
    @see: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
    """

    # config injection
    def __init__(self, in_channels, out_channels,
                 kernel_size=config.conv_kernel, stride=config.conv_stride,
                 padding=config.conv_padding, bias=config.conv_bias,
                 dropout_p=config.conv_dropout_p):
        super().__init__()

        # depthwise: "out_channels" and "groups" is same as "in_channels"
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=bias, groups=in_channels)
        # pointwise: kernel_size=1
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn = nn.BatchNorm2d(out_channels)

        nn.init.xavier_uniform_(self.depthwise.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        # Having no ReLU is advised by Xception paper (Chollet, 2017)
        x = self.dropout(x)
        x = self.bn(x)
        return x
