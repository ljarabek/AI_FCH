import os
from constants import *
from data.CSV import get_csv
from data.sumniki import popravi_sumnike
from data.flist_to_array import flist_to_array

PET_paths = list()
CT_paths = list()
for root, dirs, files in os.walk(images_path_sick):
    for dir in dirs:
        # print(dir)
        if "4mm" in dir or "4_mm" in dir:
            PET_paths.append(os.path.join(root, dir))

        if "CT" in dir:
            CT_paths.append(os.path.join(root, dir))

#print(len(CT_paths))
#print(len(PET_paths))

csv = get_csv()

cnt = 0
cnt_ = 0
for p in csv:
    patient = csv[p]
    csv[p]["cts"] = list()
    csv[p]['pets'] = list()
    # print(patient)
    for CT in CT_paths:
        name = popravi_sumnike(CT.lower())
        if name.find(patient['priimek'].lower()) != -1:
            if name.find(patient['ime'].lower()) != -1:
                if CT not in csv[p]["cts"]:
                    csv[p]["cts"].append(CT)
    for PET in PET_paths:
        name = popravi_sumnike(PET.lower())
        if name.find(patient['priimek'].lower()) != -1:
            if name.find(patient['ime'].lower()) != -1:
                if PET not in csv[p]['pets']:
                    csv[p]['pets'].append(PET)

def csv_with_fnames():
    return csv


def get_flist_from_folder(folder: str) -> list:
    otpt = list()
    for root, dirs, files in os.walk(folder):
        for f in files:
            ap=True
            fpath = os.path.join(root, f)
            if len(otpt)==0:otpt.append(fpath)
            for f in otpt:
                if fpath[-9:] in f:  # fpath ends in -0001.dcm --> so it removes duplicates...
                    ap=False
            if ap:
                otpt.append(fpath)


    return otpt

print(get_flist_from_folder('C:/Users/LeonE/Desktop/PREŠERNOVA/AI FCH/DICOM_all4mm\\Kamensek_Andrej\\Pet_Choline_Obscitnica_2Fazalm_(Adult) - 2\\4_mm_6')
      )

x = flist_to_array(get_flist_from_folder('C:/Users/LeonE/Desktop/PREŠERNOVA/AI FCH/DICOM_all4mm\\Kamensek_Andrej\\Pet_Choline_Obscitnica_2Fazalm_(Adult) - 2\\AC_CT_Obscitnica_2'))
import matplotlib.pyplot as plt

plt.hist(x.ravel(), bins=20)
plt.show()

### !!! BAD DATA !!!
for p in csv:
    patient = csv[p]
    if patient['cts'] != list(): cnt += 1
    if patient['pets'] != list(): cnt_ += 1
    #print(patient['priimek'], patient['ime'])
    #print(patient['cts'])
    #print(patient['pets'])





#print(cnt)
#print(cnt_)

# print(CT_paths[0])
