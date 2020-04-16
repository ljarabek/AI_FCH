import torch
from torch.utils.data import Dataset
from data.clean_folders import get_flist_from_folder, csv_with_fnames
from argparse import ArgumentParser
from scipy.interpolate import RectBivariateSpline
from constants import *
import os
import numpy as np
import io
import csv
import re
from constants import *

import SimpleITK as sitk

csvf = csv_with_fnames()

master_list = list()

# MAKE MASTER LIST --> of dictionaries


DATASET_IN_RAM = True


for new_id, old_id in enumerate(csvf):
    if csvf[old_id]['cts'] == [] or csvf[old_id]['pets'] == []:
        continue

    pet = get_flist_from_folder(csvf[old_id]['pets'][0])
    ct = get_flist_from_folder(csvf[old_id]['cts'][0])

    petl = list()
    ctl = list()

    for i in range(56):
        petl.append("")
        ctl.append("")
    for entry_ct in ct:
        index = int(entry_ct[-8:-4]) - 1  # konec... 0235-0019.dcm
        ctl[index] = entry_ct
    for entry_pet in pet:
        index = int(entry_pet[-8:-4]) - 1
        petl[index] = entry_pet
    if csvf[old_id]['histo'] in legal_labels:
        entry = {
            'pets': petl,
            'cts': ctl,
            'label': csvf[old_id]['histo'],
            'full_entry': csvf[old_id]
        }

        master_list.append(entry)

for aaa in os.listdir(images_path_healthy):
    rt = os.path.join(images_path_healthy, aaa)
    ct = list()
    pet = list()
    cti = 0
    peti = 0
    for fname in os.listdir(rt):
        if fname.find("CT") != -1:
            if cti == 0:
                cti = fname[:fname.find("CT") + 5:]
                ct.append(os.path.join(rt, fname))
            elif fname.startswith(cti):
                ct.append(os.path.join(rt, fname))
        if fname.find("PT") != -1:
            if peti == 0:
                peti = fname[:fname.find("PT.") + 5:]
                pet.append(os.path.join(rt, fname))
            elif fname.startswith(peti):
                pet.append(os.path.join(rt, fname))
    if len(ct) == 56 and len(pet) == 56: # nimajo vsi zdravi 56 rezin... eni majo 74
        ctl, petl = list(), list()
        for i in range(56):
            ctl.append("")
            petl.append("")
        for entry_ct in ct:
            r = re.findall(r'(\.CT\.[0-9]*\.)([0-9]*)', entry_ct)[0][1]
            r = int(r)
            ctl[r - 1] = entry_ct
        for entry_pet in pet:
            r = re.findall(r'(\.PT\.[0-9]*\.)([0-9]*)', entry_pet)[0][1]
            r = int(r)
            petl[r - 1] = entry_pet
        entry = {
            'cts': ctl,
            'pets': petl,
            'label': 'normal'
        }
        master_list.append(entry)
    # break

for pat in master_list:  # one-hot ENCODING
    l = pat['label']
    label = np.zeros(len(label_list), dtype=np.float32)
    for il, l_e in enumerate(label_list):
        if l in l_e:
            label[il] = 1.0
    pat['label'] = label


def interpolate(image, new_size: tuple):
    coordinates_1 = np.arange(0, image.shape[0], 1)
    coordinates_2 = np.arange(0, image.shape[1], 1)
    indices = (np.linspace(0, image.shape[0], new_size[0]), np.linspace(0, image.shape[1], new_size[1]))
    # print(indices[0].shape)

    spline = RectBivariateSpline(coordinates_1, coordinates_2, image)

    image = np.array(spline(indices[0], indices[1], grid=True))

    return image


class PET_CT(Dataset):
    def __init__(self, master_list_=master_list, **kwargs):
        super(PET_CT, self).__init__()
        self.master_list = master_list_
        # self.ct_mean = -930.6567472957429
        # self.ct_std = 262.7359055024721
        # self.pet_mean = 48.90856599131051
        # self.pet_std = 232.91971335374726
        self.memlist= list()
        if DATASET_IN_RAM:
            for i in range(self.__len__()):
                self.memlist.append(self.__getitem__(i))



    def __getitem__(self, idx):
        if DATASET_IN_RAM and len(self.memlist) == self.__len__():
            return self.memlist[idx]
        ctl = self.master_list[idx]['cts']
        petl = self.master_list[idx]['pets']
        label = self.master_list[idx]['label']

        ct = list()
        pet = list()
        for fname in ctl:
            image = sitk.ReadImage(fname)
            arr = sitk.GetArrayFromImage(image)[0]
            ct.append(arr)
        ct = np.array(ct)
        for fname in petl:
            image = sitk.ReadImage(fname)
            arr = sitk.GetArrayFromImage(image)[0]
            if arr.shape == (400, 400):
                arr = arr[::2, ::2]
            arr = interpolate(arr, (ct.shape[1], ct.shape[2]))
            arr[arr < 0] = 0.0  # before was 0.0 minimum
            pet.append(arr)

        ct = np.array(ct, dtype=np.float)[:, x_0 - x_r: x_0 + x_r, y_0 - y_r: y_0 + y_r]
        pet = np.array(pet, dtype=np.float)[:, x_0 - x_r: x_0 + x_r, y_0 - y_r: y_0 + y_r]

        ct = (ct - ct_mean) / ct_std
        pet = (pet - pet_mean) / pet_std

        ct = np.expand_dims(ct, -1)
        pet = np.expand_dims(pet, -1)

        merged = np.concatenate([ct, pet], -1)

        merged = np.transpose(merged)
        ct = np.transpose(ct)
        pet = np.transpose(pet)
        # print(label)
        return ct, pet, merged, label

    def __len__(self):
        return len(self.master_list)
