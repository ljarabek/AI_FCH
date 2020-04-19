from models.unet import UNet3D
import torch.nn as nn
import torch.functional as F
from resnets.resnet import resnet10
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch


class MyModel(nn.Module):
    def __init__(self, num_classes=5, **kwargs):
        super(MyModel, self).__init__()
        self.num_classes = num_classes

        self.unet = UNet3D(in_channel=2, n_classes=1)  # nima sigmoida na koncu!
        self.classifier = resnet10(num_classes=self.num_classes, activation="softmax")
        self.activation = nn.Tanh()  # nn.Sigmoid()

    def forward(self, x):
        # for i in range(23, 32, 1):
        #    slice_to_show = i
        #    plt.imshow(x_[0, 0, slice_to_show], cmap="Greys_r")  # 0 je CT
        #    plt.imshow(x_[0, 1, slice_to_show], alpha=0.3)
        #    plt.show()
        #    plt.close("all")

        x1 = self.unet(x)
        x1 = self.activation(x1) + 1.
        x_ = x1.cpu().detach().numpy()
        plt.imshow(x_[0, 0, 0])
        plt.show()
        ones = torch.ones_like(x1)
        x1 = torch.cat([ones, x1], 1)  # concatenate along channel dimension, where

        x_input = x1 * x  # with modified PET
        x_ = x_input.cpu().detach().numpy()
        plt.imshow(x_[0,1,0])
        plt.show()
        # slice_to_show = 5
        # plt.imshow(x_[0, 0, slice_to_show], cmap="Greys_r")  # 0 je CT
        # plt.imshow(x_[0, 1, slice_to_show], alpha=0.3)
        # plt.show()
        # plt.close("all")
        classified = self.classifier(x_input)
        return classified#, ()
