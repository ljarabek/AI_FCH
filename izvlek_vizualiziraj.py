import numpy as np
import os
#from multi_slice_viewer.multi_slice_viewer import seg_viewer
from multi_slice_viewer.double_viewer import seg_viewer

hista_ = np.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/outputs_/histologija_vseh.npy")

for i, hista in enumerate(hista_):
    i += 1
    att = np.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/outputs_/att_%s.npy" % i)
    input = np.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/outputs_/input_%s.npy" % i)
    output = np.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/outputs_/output_%s.npy" % i)
    weighted = np.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/outputs_/weighted_%s.npy" % i)
    print(i, hista[1])
    seg_viewer(input[1, 0], input[1, 1], input[1, 0], att[1, 0], input[1, 0], weighted[1, 1])

    #seg_viewer(weighted[0, 0], weighted[0, 0])
