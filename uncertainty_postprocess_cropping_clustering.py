import pickle
import os
from multi_slice_viewer.multi_slice_viewer import seg_viewer
import numpy as np

pickles_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH/runs_unceirtanty_model/pickles"
pkls = list()
for f in os.listdir(pickles_dir):
    pkls.append(os.path.join(pickles_dir, f))


def getpkl(f: str):
    with open(f, "rb") as file:
        return pickle.load(file)


# print(getpkl(pkls[0]))

for p in pkls:
    unpickled = getpkl(p)

    CT = unpickled['CT']
    PET = unpickled['PET']
    output = unpickled['output']
    zeros = np.zeros_like(CT)
    # print(output.shape)
    PET_T = np.maximum(zeros, PET - output[2])  # transformed PET with output to only see higher bound upliers

    print(unpickled['info'])
    seg_viewer(CT, PET_T)
    #seg_viewer(CT, PET)



    #break
