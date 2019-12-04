# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pydicom
import os
import matplotlib.pyplot as plt
import numpy as np


#class Patient():
#    def __init__(self, CT_2:list, CT_4:list, PET:list):
        

CT_2 = list()
CT_4 = list()
PET = list()

patient_root = "F:\PREŠERNOVA\PREŠERNOVA_ZDRAVI\ArZa_20191028_181946"
for root, dirs, files in os.walk(patient_root):
    for f in files:
        if "CT.2" in f:
            CT_2.append(os.path.join(patient_root,f))
        if "CT.4" in f:
            CT_4.append(os.path.join(patient_root,f))
        if "PT." in f:
            PET.append(os.path.join(patient_root,f))

file = pydicom.read_file(PET[0])

#print(vars(file))

data = np.array(file.pixel_array)
print(data.shape)

plt.imshow(data)
plt.show()
