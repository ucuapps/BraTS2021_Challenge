import torch
import torch.nn as nn
from collections import OrderedDict


class ChannelSeparableConv3d(nn.Sequential):
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride=1, bias=False):
        super(ChannelSeparableConv3d, self).__init__(OrderedDict([
            ('channelwise',
             nn.Conv3d(nin, nin, kernel_size=kernel_size, padding=padding, stride=stride, groups=nin, bias=bias)),
            ('pointwise', nn.Conv3d(nin, nout, kernel_size=1, bias=bias))
        ]))


class ChannelSeparableConvTranspose3d(nn.Sequential):
    def __init__(self, nin, nout, kernel_size=2, padding=0, stride=2, bias=False):
        super(ChannelSeparableConvTranspose3d, self).__init__(OrderedDict([
            ('channelwise',
             nn.ConvTranspose3d(nin, nin, kernel_size=kernel_size, padding=padding, stride=stride, groups=nin,
                                bias=bias)),
            ('pointwise', nn.Conv3d(nin, nout, kernel_size=1, bias=bias))
        ]))


if __name__ == '__main__':
    x = torch.randn(10, 25, 10, 10, 10)
    model = ChannelSeparableConv3d(25, 36, 3, 1, True)
    y = model(x)
    print(y.shape)
