import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import random
import torchvision.transforms as transforms

# "When more images are available we select the 9 clearest images according to the masks" - DeepSUM
# I am just going to randomly select 9, partly because its easier, partly because maybe this will help the model be more
# resilient to cloud cover when testing

# "As a
# preprocessing step, all LR images are clipped to 2^14 - 1
# since corrupted pixels with large values occur in the LR images
# throughout the PROBA-V dataset."

# [b,9,W,H,1] theirs
# [b,1,9,W,H] ours


class TrainNIRDataset(torch.utils.data.Dataset):
    """Some Satellite Shit."""

    def __init__(self, upsample=True):
        self.upsample = upsample
        if self.upsample:
            self.transform = transforms.Compose([transforms.Resize((384, 384), interpolation=Image.BICUBIC), transforms.ToTensor()])
            self.size = 384
        else:
            self.transform = transforms.ToTensor()
            self.size = 128

    def __len__(self):
        return 566 * 100

    def __getitem__(self, idx):
        start_index = 594
        img_set_num = start_index + idx // 100
        img_set_filename = '0' + str(img_set_num) if img_set_num < 1000 else str(img_set_num)

        imgs_path = 'data/train/NIR/imgset' + str(img_set_filename)

        num_LR = len([name for name in os.listdir(imgs_path)]) // 2 - 1
        selected_LR = np.random.choice(num_LR, 9, replace=False)

        #if upsample == true:
        #    LR_imgs[]

        #    do stuff to get blocks, get 100 for any scene -> 1 HR, 9 LR -> getitem gets 1 scene, return 100 patches


        #size: torchee tensor: (9, 384, 384)
        LR_imgs = torch.cat([self.transform(Image.open(imgs_path + '/LR{}.png'.format(('00' + str(val)) if val < 10 else ('0' + str(val)))))
                            for val in selected_LR]).resize_((1, 9, self.size, self.size))


        #size: torcheee tensor: (1, 384, 384)
        HR_img = torch.Tensor(np.array(Image.open(imgs_path + '/HR.png')))

        if self.upsample: # DONG
            thisFrame = random.randint(0, 288)
            LR_imgs = LR_imgs[:, :, thisFrame:thisFrame+96, thisFrame:thisFrame+96]
            HR_img = HR_img[thisFrame:thisFrame+96, thisFrame:thisFrame+96]

        sample = {'LR': torch.clamp(LR_imgs, min=0, max=2**14-1), 'HR': HR_img.view(1, 96, 96)}
        return sample

class TestNIRDataset(torch.utils.data.Dataset):
    """Some Satellite Shit."""

    def __init__(self, upsample=True):
        self.upsample = upsample
        if self.upsample:
            self.transform = transforms.Compose([transforms.Resize((384, 384), interpolation=Image.BICUBIC), transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return 144

    def __getitem__(self, idx):

        start_index = 1306
        img_set_num = start_index + idx
        img_set_filename = '0' + str(img_set_num) if img_set_num < 1000 else str(img_set_num)
        imgs_path = 'data/test/NIR/imgset' + str(img_set_filename)

        num_LR = len([name for name in os.listdir(imgs_path)]) // 2
        selected_LR = np.random.choice(num_LR, 9, replace=False)
        LR_imgs = torch.cat([self.transform(Image.open(imgs_path + '/LR{}.png'.format(('00' + str(val)) if val < 10 else ('0' + str(val)))))
                            for val in selected_LR]).resize_((1, 9, self.size, self.size))

        sample = {'LR': torch.clamp(LR_imgs, min=0, max=2**14-1)}
        return sample

class TrainREDDataset(torch.utils.data.Dataset):
    """Some Satellite Shit."""

    def __init__(self, upsample=True):
        self.upsample = upsample
        if self.upsample:
            self.transform = transforms.Compose([transforms.Resize((384, 384), interpolation=Image.BICUBIC), transforms.ToTensor()])
            self.size = 384
        else:
            self.transform = transforms.ToTensor()
            self.size = 128

    def __len__(self):
        return 594

    def __getitem__(self, idx):

        start_index = 0
        img_set_num = start_index + idx
        img_set_filename = str(img_set_num)
        while len(img_set_filename) < 4:
            img_set_filename = '0' + img_set_filename
        imgs_path = 'data/train/RED/imgset' + str(img_set_filename)

        num_LR = len([name for name in os.listdir(imgs_path)]) // 2 - 1
        selected_LR = np.random.choice(num_LR, 9, replace=False)
        LR_imgs = torch.cat([self.transform(Image.open(imgs_path + '/LR{}.png'.format(('00' + str(val)) if val < 10 else ('0' + str(val)))))
                            for val in selected_LR]).resize_((1, 9, self.size, self.size))


        #size: torcheee tensor: (1, 384, 384)
        HR_img = torch.Tensor(np.array(Image.open(imgs_path + '/HR.png')))

        if self.upsample:
            thisFrame = random.randint(0, 288)
            LR_imgs = LR_imgs[:, :, thisFrame:thisFrame+96, thisFrame:thisFrame+96]
            HR_img = HR_img[thisFrame:thisFrame+96, thisFrame:thisFrame+96]

        sample = {'LR': torch.clamp(LR_imgs, min=0, max=2**14-1).view(9, 96, 96), 'HR': HR_img.view(1, 96, 96)}
        return sample


class TestREDDataset(torch.utils.data.Dataset):
    """Some Satellite Shit."""

    def __init__(self, upsample=True):
        self.upsample = upsample
        if self.upsample:
            self.transform = transforms.Compose([transforms.Resize((384, 384), interpolation=Image.BICUBIC), transforms.ToTensor()])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return 146

    def __getitem__(self, idx):

        start_index = 1160
        img_set_num = start_index + idx
        img_set_filename = '0' + str(img_set_num) if img_set_num < 1000 else str(img_set_num)
        imgs_path = 'data/test/RED/imgset' + str(img_set_filename)

        num_LR = len([name for name in os.listdir(imgs_path)]) // 2
        selected_LR = np.random.choice(num_LR, 9, replace=False)
        LR_imgs = torch.cat([self.transform(Image.open(imgs_path + '/LR{}.png'.format(('00' + str(val)) if val < 10 else ('0' + str(val)))))
                            for val in selected_LR]).resize_((1, 9, self.size, self.size))

        sample = {'LR': torch.clamp(LR_imgs, min=0, max=2**14-1)}
        return sample
