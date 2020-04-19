import pickle
from sklearn import svm, datasets
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

metrics = {
    'fname': "classifications_best_resnet10.pkl",  # best_resnet10.pkl"
    "confusion_matrix": dict(),
    'AUC':dict()
}


from datetime import datetime
import os
import json

fname = metrics['fname']
with open(fname, "rb") as f:
    classifications = pickle.load(f)
matplotlib.use('Qt5Agg')
print(fname)


# AUC CODE
today = datetime.now()
eval_dir = "./runs_evaluations/" + today.strftime('%Y%m%d%h%m%s') + "/"
os.makedirs(eval_dir, exist_ok=True)

for label in classifications['truth']:
    metrics['confusion_matrix'][label] = {'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0}
for label in classifications['truth']:
    # roc_curve(y_true = classifications['truth'][label], y_score=classifications['pred'][label])
    score = roc_auc_score(y_true=classifications['truth'][label], y_score=classifications['pred'][label])
    print("AUC %s %s" % (label, score))
    metrics['AUC'][label]=score
    lo = roc_curve(y_true=classifications['truth'][label], y_score=classifications['pred'][label])
    print(np.array(lo).shape)
    plt.plot(lo[0], lo[1])  # PLOT ROC CURVE
    plt.title(label)
    plt.savefig(eval_dir + label + "_ROC.png")
    plt.close("all")

# print(metrics['confusion_matrix'])
for ex_i, truth in enumerate(classifications['truth']['DS']):
    preds = [x for x in [classifications['pred'][label_][ex_i] for label_ in classifications['pred']]]
    truths = [x for x in [classifications['truth'][label_][ex_i] for label_ in classifications['truth']]]
    truth_no = truths.index(max(truths))

    preds_ = list()

    pred_no = preds.index(max(preds))
    preds[pred_no] = 0

    treshold = 0.2

    preds_.append(pred_no)
    preds_.append(preds.index(max(preds)))
    for il, label in enumerate(classifications['truth']):
        # gremo po vsakem labelu
        if truth_no != il:  # če je ta primer drug label od tistega kerega gledamo
            if il == pred_no:  # če smo predvidli da je tisti, kerega gledamo
                metrics['confusion_matrix'][label]['FP'] += 1
            if il != pred_no:  # če nismo predvidli da je tisti, kerega gledamo
                metrics['confusion_matrix'][label]['TN'] += 1

        if truth_no == il:  # če je ta primer isti label katerega opazujemo
            if il in preds_:
                metrics['confusion_matrix'][label]['TP'] += 1
            if il not in preds_:
                metrics['confusion_matrix'][label]['FN'] += 1

from pprint import pprint

pprint(metrics['confusion_matrix'])

dataset_len = len(classifications['truth']['DS'])

for label in metrics['confusion_matrix']:
    metrics['confusion_matrix'][label]["accuracy"] = (metrics['confusion_matrix'][label]['TP'] +
                                                      metrics['confusion_matrix'][label]['TN']) \
                                                     / dataset_len
    # Technically the raw prediction accuracy of the model is defined as
    # (TruePositives + TrueNegatives)/SampleSize.
    metrics['confusion_matrix'][label]["error_rate"] = 1 - metrics['confusion_matrix'][label]["accuracy"]
    try:
        metrics['confusion_matrix'][label]['precision'] = metrics['confusion_matrix'][label]['TP'] / (
                metrics['confusion_matrix'][label]['TP'] + metrics['confusion_matrix'][label][
            'FP'])  # PPV = Precision = TruePositives/(TruePositives + FalsePositives)
    except ZeroDivisionError:
        metrics['confusion_matrix'][label]['precision'] = 0.0

    metrics['confusion_matrix'][label]['sensitivity'] = metrics['confusion_matrix'][label]['TP'] / (
            metrics['confusion_matrix'][label]['TP'] + metrics['confusion_matrix'][label][
        'FN'])  # Sensitivity = TruePositives/(TruePositives + FalseNegatives

    metrics['confusion_matrix'][label]['specificity'] = metrics['confusion_matrix'][label]['TN'] / (
            metrics['confusion_matrix'][label]['TN'] + metrics['confusion_matrix'][label][
        'FN'])  # Specificity = TrueNegatives/(TrueNegatives + FalseNegatives)

    metrics['confusion_matrix'][label]['false_positive_rate'] = 1 - metrics['confusion_matrix'][label][
        'specificity']  # 1-specificity

    metrics['confusion_matrix'][label]['negative_predictive_value'] = metrics['confusion_matrix'][label]['TN'] / (
            metrics['confusion_matrix'][label]['TN'] + metrics['confusion_matrix'][label]['FN'])

    metrics['confusion_matrix'][label]['f1-score'] = 2 * metrics['confusion_matrix'][label]['TP'] / (
            2 * metrics['confusion_matrix'][label]['TP'] + metrics['confusion_matrix'][label]['FP'] +
            metrics['confusion_matrix'][label]['FN'])  # source : https://en.wikipedia.org/wiki/Precision_and_recall

    # TODO maybe: Kappa, Mcnemar's Test P-Value, accuracy confidence interval, N0 information rate, P-Value (http://www.sthda.com/english/articles/36-classification-methods-essentials/143-evaluation-of-classification-model-accuracy-essentials/).

import json

print(metrics)
with open(eval_dir +"metrics.json", "w") as f:
    json.dump(metrics,f)
