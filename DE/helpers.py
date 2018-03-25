"""Helper functions."""
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import auc

PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


"Given prediction and truth, get tp, fp, tn, fn. "


def get_abcd(predict, truth):
    # pos > 0, neg == 0
    n = len(predict)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(n):
        if predict[i] > 0 and truth[i] > 0:
            tp += 1
        elif predict[i] > 0 and truth[i] == 0:
            fp += 1
        elif predict[i] == 0 and truth[i] == 0:
            tn += 1
        elif predict[i] == 0 and truth[i] > 0:
            fn += 1
    return tp, fp, tn, fn


"Given TP, FP, TN, FN, get all the other metrics. "


def get_performance(metrics):
    tp, fp, tn, fn = metrics
    pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
    fpr = 1 - spec
    npv = 1.0 * tn / (tn + fn) if (tn + fn) != 0 else 0
    acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
    return [round(x, 3) for x in [pre, rec, spec, fpr, npv, acc, f1]]


"Given the general metrics, return the score got by the specific criteria."


def get_score(criteria, metrics):   # The smaller the better
    tp, fp, tn, fn = metrics
    pre, rec, spec, fpr, npv, acc, f1 = get_performance([tp, fp, tn, fn])
    all_metrics = [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]
    if criteria == "Accuracy":
        score = -all_metrics[-ACC]
    elif criteria == "Dist2Heaven":
        score = all_metrics[-FPR] ** 2 + (1 - all_metrics[-REC]) ** 2
        score = math.sqrt(score) / math.sqrt(2)
    elif criteria == "Gini":
        p1 = all_metrics[-PRE]  # target == 1 for the positive split
        p0 = 1 - all_metrics[-NPV]  # target == 1 for the negative split
        score = 1 - p0 ** 2 - p1 ** 2
    else:  # Information Gain
        P, N = all_metrics[0] + all_metrics[3], all_metrics[1] + all_metrics[2]
        p = 1.0 * P / (P + N) if P + N > 0 else 0  # before the split
        p1 = all_metrics[-PRE]  # the positive part of the split
        p0 = 1 - all_metrics[-NPV]  # the negative part of the split
        I, I0, I1 = (-x * np.log2(x) if x != 0 else 0 for x in (p, p0, p1))
        I01 = p * I1 + (1 - p) * I0
        score = -(I - I01)  # the smaller the better.
    return round(score, 3)


def subtotal(x):
    xx = [0]
    for i, t in enumerate(x):
        xx += [xx[-1] + t]
    return xx[1:]


def get_recall(true):
    total_true = float(len([i for i in true if i == 1]))
    hit = 0.0
    recall = []
    for i in xrange(len(true)):
        if true[i] == 1:
            hit += 1
        recall += [hit / total_true if total_true else 0.0]
    return recall


def cut_position(pos, neg, percentage=0):
    return int(pos.shape[0] * percentage / 100), int(neg.shape[0] * percentage / 100)
    #return int(pos.shape[0] * percentage / 100), int(neg.shape[0] * percentage / 100)

def divide_train_test(pos, neg, cut_pos, cut_neg):
    data_train = pd.concat([pos.iloc[:cut_pos,:], neg.iloc[:cut_neg,:]],ignore_index=True)
    data_test = pd.concat([pos.iloc[cut_pos:,:], neg.iloc[cut_neg:,:]], ignore_index=True)
    return data_train, data_test


def split_two(corpus):
    pos = corpus[corpus['bug']==1]
    neg = corpus[corpus['bug'] != 1]
    return {'pos': pos, 'neg': neg}


def get_auc(data):
    """The smaller the better"""
    if len(data) == 1:
        return 0
    x_sum = float(sum(data['loc']))
    if x_sum > 0:
        x = data['loc'].apply(lambda t: t / x_sum)
    else:
        x = data['loc']
    xx = subtotal(x)
    yy = get_recall(data['bug'].values)
    try:
        ret = -round(auc(xx, yy), 3)
    except:
        print "?"
        ret = 0
    return ret



