from __future__ import print_function, division

import sys
import inspect

sys.dont_write_bytecode = True
from ABCD import ABCD
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scores import *
from helpers import *
import pandas as pd
import new_fft

recall, precision, specificity, accuracy, f1, g, f2, d2h = 8, 7, 6, 5, 4, 3, 2, 1

def DT(train_data,train_labels,test_data):
    model = DecisionTreeClassifier(criterion='entropy').fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return prediction

def KNN(train_data,train_labels,test_data):
    model = KNeighborsClassifier(n_neighbors=8,n_jobs=-1).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction

def LR(train_data,train_labels,test_data):
    model = LogisticRegression().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction

def NB(train_data,train_labels,test_data):
    model = GaussianNB().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction

def RF(train_data,train_labels,test_data):
    model = RandomForestClassifier(criterion='entropy').fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction

def SVM(train_data,train_labels,test_data):
    model = LinearSVC().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction

def FFT(data_train, data_test):
    model = new_fft.FFT(max_level=5, max_depth=4, medianTop=1)
    model.ignore = {"name", "version", 'name.1', 'prediction'}
    model.target =
    model.train = data_train
    model.test = data_test
    return model



def evaluation(measure, prediction, test_labels, test_data):
    abcd = ABCD(before=test_labels, after=prediction)
    stats = np.array([j.stats() for j in abcd()])
    labels = list(set(test_labels))
    if labels[0] == 0:
        target_label = 1
    else:
        target_label = 0

    if measure == "accuracy":
        return stats[target_label][-accuracy]
    if measure == "recall":
        return stats[target_label][-recall]
    if measure == "precision":
        return stats[target_label][-precision]
    if measure == "specificity":
        return stats[target_label][-specificity]
    if measure == "f1":
        return stats[target_label][-f1]
    if measure == "f2":
        return stats[target_label][-f2]
    if measure == "d2h":
        return stats[target_label][-d2h]
    if measure == "g":
        return stats[target_label][-g]
    if measure == "popt20":
        df1 = pd.DataFrame(prediction, columns=["prediction"])
        df2 = pd.concat([test_data, df1], axis=1)
        return get_popt(df2)


def fft_process(*x):
    l = np.asarray(x)
    model = l[1]
    criteria = l[2]
    data = l[3]

    shuffle_indices = np.random.permutation(np.arange(data.shape[0]))
    shuffled_data = data.ix[shuffle_indices]
    msk = int(len(shuffled_data)*0.8)
    data_train, data_test = data.iloc[:msk], data.iloc[msk:]
    return fft_eval(model, data_train, data_test, criteria, l[0])


def fft_eval(model, data_train, data_test, criteria, params):
    fft = model(data_train, data_test)
    fft.criteria = criteria
    fft.build_trees()  # build and get performance on TRAIN data
    t_id = fft.find_best_tree()  # find the best tree on TRAIN data
    fft.eval_trees()
    fft.print_tree(t_id)
    print(fft.results[criteria])
    return fft.results[criteria]





#discritization: mean, median, percentile_chop
#build rules that sample
