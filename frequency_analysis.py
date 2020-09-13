import pickle
import numpy as np




"""master_pkl_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH/data/master.pkl"
with open(master_pkl_dir, "rb") as f:
    # 1/0
    settings, master_list = pickle.load(f)

# print(master_list[0])

locs = list()
Ca = list()
fosfat = list()
iPTH = list()
starost = list()
for pat in master_list:
    locs.append(pat["histo_lokacija"])
    try:
        Ca.append(float(pat['Ca']))
    except:
        pass
    try:
        fosfat.append(float(pat['fosfat']))
    except:
        pass
    try:
        iPTH.append(float(pat['iPTH']))
    except:
        pass
    try:
        starost.append(int(pat['starost']))
    except:
        pass

print("Ca %s %s" % (np.mean(Ca), np.std(Ca)))
print("fosfat %s %s" % (np.min(fosfat), np.max(fosfat)))
print("starost %s %s" % (np.min(starost), np.max(starost)))
print("iPTH %s %s" % (np.mean(iPTH), np.std(iPTH)))

import matplotlib.pyplot as plt

loc_no = dict()
for l in locs:
    if l not in loc_no:
        loc_no[l] = 1
    else:
        loc_no[l] += 1
print(loc_no
      )  # plus 4 ektopični!

import matplotlib.pyplot as plt
import numpy as np
for category in categories:
    try:
        cat = np.array(categories[category])
    except:
        cat = categories[category]
        pass
    if category is 'pets' or category is 'cts' or category is 'priimek' or category is 'ime':
        continue
    plt.hist(cat, bins=20)
    plt.title(category)
    plt.show()

zaenkrat LS LZ DS!!"""
