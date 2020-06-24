import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def plot_images(predic_batch, hr_batch, num_imgs_shown=4):
    '''
    Input:
    - predic_batch: batch of low res images of size (-1, 1, W, H)
    - hr_batch: batch of high res images of size (-1, 1, W, H)
    - num_imgs_shown: number of images to plot in tensorboard

    Output:
        - fig: a matplotlib figure containing the image which will be written to
        tensorboard
    '''

    # TODO:

    fig, ax = plt.subplots(nrows=2, ncols=num_imgs_shown)

    for i in range(0, num_imgs_shown):
        img = predic_batch[i]
        img2 = hr_batch[i]
        ax[0, i].axis('off')
        ax[0, i].imshow(img)
        ax[1, i].axis('off')
        ax[1, i].imshow(img2)
    
    plt.show()



    

    return fig
