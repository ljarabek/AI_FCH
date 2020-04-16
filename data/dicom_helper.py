import numpy as np
import pydicom
import os
from collections import defaultdict


def flist_to_array(x):
    lst = list()
    for file in x:
        file = pydicom.read_file(file)
        x = file.pixel_array
        x = np.array(x, dtype=np.float32)
        lst.append(x)
    x = np.array(lst, dtype=np.float32)
    return x


def match_healthy_pet_cts(folder):
    cts_ = list()
    pets_ = list()
    studies = defaultdict(list)
    for file_ in os.listdir(folder):

        # print(file_)
        try:
            file = pydicom.read_file(os.path.join(folder, file_))
        except:
            print("fail at file%s"%os.path.join(folder, file_))
            return
        study_id = file[0x0020000E].value

        study_id = file[0x0020000D].value  # TE SO ISTE, ČE je ista študija!!
        #print(study_id)

        size_ = file[0x0008103E].value  # NE VZET ITERATIVE!!!
        #print(size_)
        if "CT" in size_ and "iter" not in str(size_).lower():
            studies[study_id + "_CT"].append(file_)
            cts_.append(file_)
        if "4" in size_:
            studies[study_id + "_PET"].append(file_)
            pets_.append(file_)
    print(folder)
    for key in studies:
        print(key)
        print(len(studies[key]))
    #print(studies)
    # print(pets_)
    # print(cts_)
    # if len(cts_) ==0:
    #    print(folder)
    #    print("MISSING CTS")
    # if len(pets_) == 0:
    #    print(folder)
    #    print("MISSING PETS")
    #    #print("slice : %s"%file[0x00201041].value)
    # if "PT" in file_:
    #    print(study_id)
