import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
from data import *

hr_size = (96,96)
nsf = 64 # number of sisrnet filters

# [b,9,W,H,1] theirs (DeepSUM in TensorFlow)
# [b,1,9,W,H] ours, input size to SISRNet

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

        # [b,64,9,W,H] output size

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

class SISRNet_pretrained(nn.Module):
    def __init__(self):
        super(SISRNet_pretrained, self).__init__()
        self.sisrnet = SISRNet()
        self.projection = nn.Sequential(...)


        # [b,64,9,W,H] output size

    def forward(self, input):

        # D stays the same throughout, so do H and W (padding=1 with "reflect", kernel=3)
        # for now im using InstanceNorm3d, not sure if its correct yet

        # use conv3d for these 2d convolutions with size 1 depth kernel so that
        # the 2d weights are shared across the 9 LR images
        input = self.projection(self.sisrnet(input))
        return input

import numpy as np
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

model = SISRNet()
# print(model, 'with', num_params(model), 'parameters')

dataloader = torch.utils.data.DataLoader(TrainNIRDataset(), batch_size = 1)

batch = next(iter(dataloader))
img = batch["HR"].view(1, 1, 384, 384).float()
# plt.imshow(img.view(384, 384), cmap='gray')
print(img.shape)
# plt.show()

# filter = torch.Tensor([[[[0, 0, 0],[0, 1, 0],[0, 0, 0]]]]).float()
filter = torch.zeros((1,1,11,11)).float()
filter[0, 0, 0, 0] = 1
print(filter.shape)
apply_kernel = lambda img: F.conv2d(img, filter, stride=1, padding=5)

img2 = apply_kernel(img)
print(img2.shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img.view(384, 384), cmap='gray')
axes[1].imshow(img2.view(384, 384), cmap='gray')
plt.show()
