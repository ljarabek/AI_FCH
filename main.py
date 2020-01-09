import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

from data.dataset import PET_CT
import torch

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Run:
    def __init__(self, args):
        a = 0
        self.params = list()
        # return

    def get_dataloader(self):
        a = 0

    def forward(self):
        a = 0


if __name__ == "__main__":
    ds = PET_CT()
    master_ct = list()
    master_pet = list()

    ct, pet, merged, label = ds[3]

    print(merged.shape)
    """for i in range(len(ds)):
        print(i)
        ct,pet,label = ds[i+20]
        master_ct.append(pet)
        if i>20:
            break
    print(np.mean(master_ct),np.std(master_ct))"""
