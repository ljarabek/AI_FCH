import numpy as np
import pydicom


def flist_to_array(x):
    lst = list()
    for file in x:
        file = pydicom.read_file(file)
        x = file.pixel_array
        x = np.array(x, dtype=np.float32)
        lst.append(x)
    x = np.array(lst, dtype=np.float32)
    return x




