import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def plot_images(model, lr_batch, hr_batch, num_imgs_shown=4):
    '''
    Input:
    - model: super res model
    - lr_batch: batch of low res images of size (-1, 1, 9, W, H)
    - hr_batch: batch of high res images of size (-1, 1, W, H)
    - num_imgs_shown: number of images to plot in tensorboard

    Output:
        - fig: a matplotlib figure containing the image which will be written to
        tensorboard
    '''

    # TODO:

    return fig
