import data.clean_folders as cf

csv = cf.csv_with_fnames()

categories = dict()
for pat in csv:
    patient = csv[pat]
    if patient['cts'] == [] or patient['pets'] == []:
        continue
    for category in patient:
        attr = patient[category]
        if category not in categories:
            categories[category] = [attr]
        else:
            categories[category].append(attr)

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
