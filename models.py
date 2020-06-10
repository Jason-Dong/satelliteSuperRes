import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

hr_size = (96,96)
nsf = 64 # number of sisrnet filters

# [b,9,W,H,1] theirs (DeepSUM in TensorFlow)
# [b,1,9,W,H] ours

class SISRNet(nn.Module):
    def __init__(self):
        super(SISRNet, self).__init__()
        self.block_1 = nn.Sequential(nn.Conv3d(1, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))

        self.block_2 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))
        self.block_3 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))
        self.block_4 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))
        self.block_5 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))
        self.block_6 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))
        self.block_7 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))
        self.block_8 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"),
                            nn.InstanceNorm3d(nsf), nn.LeakyReLU(0.01))

        # self.block_1 = nn.Sequential(nn.Conv3d(1, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_2 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_3 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_4 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_5 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_6 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_7 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))
        # self.block_8 = nn.Sequential(nn.Conv3d(nsf, nsf, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="reflect"), nn.LeakyReLU(0.01))


    def forward(self, input):

        # D stays the same throughout, so do H and W (padding=1 with "reflect", kernel=3)
        # for now im using InstanceNorm3d, not sure if its correct yet

        # use conv3d for these 2d convolutions with size 1 depth kernel so that
        # the 2d weights are shared across the 9 LR images

        input = self.block_1(input)
        input = self.block_2(input)
        input = self.block_3(input)
        input = self.block_4(input)
        input = self.block_5(input)
        input = self.block_6(input)
        input = self.block_7(input)
        input = self.block_8(input)

        return input

import numpy as np
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

model = SISRNet()
print(model, 'with', num_params(model), 'parameters')
