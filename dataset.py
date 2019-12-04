import torch
from torch.utils.data import Dataset
from argparse import ArgumentParser
from constants import *
import os
import io
import csv

import pydicom

class PET_CT(Dataset):
    def __init__(self, args):
        self.args = args



        a = 0

    def __getitem__(self):
        a=0
        return a
    def __len__(self):
        return 42


Patients_DCT = dict()

if __name__=="__main__":
    print(torch.Tensor(0))

    parser = ArgumentParser()
    parser.add_argument("--sick_images", default=sick_images_path)
    parser.add_argument("--csv_file",default=csv_path)
    args = parser.parse_args()


    names = list()
    for root, dirs, files in os.walk(args.sick_images):
        for dir in dirs:
            for name in dir.split("_"):
                names.append(name.lower())
        break
    print(names)
    csv_list = list()
    keys = list()
    with io.open(args.csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = " ", quotechar="|")

        for i, row in enumerate(reader):
            if i==0:
                keys = row
            else:
                csv_list.append(row)

    class Patient():
        def __init__(self, row, keyrow = keys):
            self.attrs = dict()





    print(csv_list[2])





                #break
        #for dir in dirs:
            #if "2 mm" in dir or "2mm" in dir:
                #print(os.path.join(root, dir))
                #pydicom.filereader.read_dataset