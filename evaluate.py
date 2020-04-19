import pickle
from sklearn import svm, datasets
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fname = "classifications_best_MyModel.pkl"
with open(fname, "rb") as f:
    classifications = pickle.load(f)
matplotlib.use('Qt5Agg')
print(fname)
for label in classifications['truth']:
    # roc_curve(y_true = classifications['truth'][label], y_score=classifications['pred'][label])
    score = roc_auc_score(y_true=classifications['truth'][label], y_score=classifications['pred'][label])
    print("AUC %s %s" % (label, score))
    lo = roc_curve(y_true=classifications['truth'][label], y_score=classifications['pred'][label])
    print(np.array(lo).shape)
    plt.plot(lo[0], lo[1])  # PLOT ROC CURVE
    plt.title(label + "_" + fname)
    plt.show()
