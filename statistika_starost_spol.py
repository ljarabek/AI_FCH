import SimpleITK as sitk
import os
from data.dataset import master_list
from datetime import datetime
import scipy.stats as stat
import numpy as np

def time(s):
    return datetime(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))


def get_info(ts):
    im = os.path.join(ts, os.listdir(ts)[0])
    image = sitk.ReadImage(im)
    date = image.GetMetaData("0008|0021")
    birthday = "19" + image.GetMetaData("0010|0030")[2:]
    age = time(date) - time(birthday)
    age = age.days // 365
    sex = image.GetMetaData('0010|0040')
    return age, sex


a, b = list(), list()
a_s, b_s = str(), str()
aage, bage = list(), list()
for s in master_list:
    # print(s.keys())
    # print(s['PET_dir'])
    age, sex = get_info(s['PET_dir'])
    loc = s['histo_lokacija']
    if loc == 'healthy':
        a.append(age)
        a_s += sex
        aage.append(age)
    else:
        b.append(age)
        b_s += sex
        bage.append(age)
print("ttest starost")
print(stat.ttest_ind(a, b))

sexa = np.array([a_s.count("M"), a_s.count("F")], dtype=float)
print("spol kontrole")
print(sexa)
sexa = sexa/np.sum(sexa)
sexb = np.array([b_s.count("M"), b_s.count("F")], dtype=float)
print("spol pacienti")
print(sexb)
sexb = sexb/np.sum(sexb)
print(sexa, sexb)
print("hikvadrat spol")
print(stat.chisquare(sexb, sexa))

print("age statistics healthy:")
print(np.mean(aage), np.std(aage))
print("age statistics patients:")
print(np.mean(bage), np.std(bage))
