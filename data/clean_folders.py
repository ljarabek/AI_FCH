import os
from constants import *
from data.CSV import get_csv
from data.sumniki import popravi_sumnike

PET_paths = list()
CT_paths = list()
for root, dirs, files in os.walk(images_path_sick):
    for dir in dirs:
        #print(dir)
        if "4mm" in dir or "4_mm" in dir:
            PET_paths.append(os.path.join(root, dir))

        if "CT" in dir:
            CT_paths.append(os.path.join(root, dir))

print(len(CT_paths))
print(len(PET_paths))

csv = get_csv()

cnt = 0
for p in csv:
    patient = csv[p]
    csv[p]["cts"] = list()
    #print(patient)
    for CT in CT_paths:
        name = popravi_sumnike(CT.lower())
        if name.find(patient['priimek'].lower()) != -1:
            if name.find(patient['ime'].lower()) != -1:
                if CT not in csv[p]["cts"]:
                    csv[p]["cts"].append(CT)
                    cnt+=1

for p in csv:
    patient = csv[p]
    print(patient)

print(cnt)

#print(CT_paths[0])