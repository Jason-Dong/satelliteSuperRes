import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

hr_size = (384,384)
nsf = 64 # number of sisrnet filters

# [b,9,W,H,1] theirs (DeepSUM in TensorFlow)
# [b,1,9,W,H] ours

class SISRNet(nn.Module):
    def __init__(self, n_blocks=8):
        super(SISRNet, self).__init__()
        self.n_blocks = 8

    def forward(self, input):

        # D stays the same throughout, so do H and W (padding=1 with "reflect", kernel=3)
        # for now im using InstanceNorm3d, not sure if its correct yet

        # use conv3d for these 2d convolutions with size 1 depth kernel so that
        # the 2d weights are shared across the 9 LR images
        input = nn.Conv3d(1, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect")(input)
        input = nn.InstanceNorm3d(input.shape[1])(input)
        input = nn.LeakyReLu(0.01)(input)
        for _ in range(self.n_blocks - 1):
            input = nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect")(input)
            input = nn.InstanceNorm3d(input.shape[1])(input)
            input = nn.LeakyReLu(0.01)(input)

        return input
