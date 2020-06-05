import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
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
        return 566

    def __getitem__(self, idx):
        start_index = 594
        img_set_num = start_index + idx
        img_set_filename = '0' + str(img_set_num) if img_set_num < 1000 else str(img_set_num)
        imgs_path = 'data/train/NIR/imgset' + str(img_set_filename)

        num_LR = len([name for name in os.listdir(imgs_path)]) // 2 - 1
        selected_LR = np.random.choice(num_LR, 9, replace=False)
        LR_imgs = torch.cat([self.transform(Image.open(imgs_path + '/LR{}.png'.format(('00' + str(val)) if val < 10 else ('0' + str(val)))))
                            for val in selected_LR]).resize_((1, 9, self.size, self.size))

        HR_img = np.array(Image.open(imgs_path + '/HR.png'))

        sample = {'LR': torch.clamp(LR_imgs, min=0, max=2**14-1), 'HR': HR_img}
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
        else:
            self.transform = transforms.ToTensor()

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

        HR_img = np.array(Image.open(imgs_path + '/HR.png'))

        sample = {'LR': torch.clamp(LR_imgs, min=0, max=2**14-1), 'HR': HR_img}
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
