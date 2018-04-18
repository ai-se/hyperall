from __future__ import print_function, division

import sys
import inspect

sys.dont_write_bytecode = True
from ABCD import ABCD
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scores import *
from helpers import *
from sklearn.model_selection import KFold

import pandas as pd

recall, precision, specificity, accuracy, f1, g, f2, d2h = 8, 7, 6, 5, 4, 3, 2, 1


def DTC(train_data,train_labels,test_data, params):
    if params:
        model = DecisionTreeClassifier(criterion=params[0], max_features=params[1], min_samples_split=params[2],
                                       min_samples_leaf=params[3], max_depth=params[4], random_state=47)
    else:
        model = DecisionTreeClassifier(random_state=47).fit(train_data, train_labels)
    model.fit(train_data, train_labels)
    prediction=model.predict(test_data)
    return prediction


def RF(train_data,train_labels,test_data, params):
    if params:
        model = RandomForestClassifier(criterion=params[0], max_features=params[1], min_samples_split=params[2],
                                       min_samples_leaf=params[3], n_estimators=params[4], random_state=47, n_jobs=4)
    else:
        model = RandomForestClassifier(random_state=47, n_jobs=4).fit(train_data, train_labels)
    model.fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction


def KNN(train_data,train_labels,test_data, params):
    if params:
        model = KNeighborsClassifier(n_neighbors=params[0], weights=params[1]).fit(train_data, train_labels)
    else:
        model = KNeighborsClassifier().fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction


def SVM(train_data,train_labels,test_data, params):
    if params:
        model = SVC(C=params[0], kernel=params[1], coef0=params[2], gamma=params[3]).fit(train_data, train_labels)
    else:
        model = SVC(random_state=47).fit(train_data, train_labels)
    prediction = model.predict(test_data)
    return prediction


def evaluation(measure, prediction, test_labels, test_data):
    abcd = ABCD(before=test_labels, after=prediction)
    stats = np.array([j.stats() for j in abcd()])
    labels = list(set(test_labels))
    if labels[0] == 0:
        target_label = 1
    else:
        target_label = 0
    try:
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
            df_test = test_data.copy()
            df_test['$<bug'] = test_labels
            df_test['prediction'] = prediction
            return get_popt(df_test)
    except:
        print("error stats")
        return 0.0


def process(*x):
    l = np.asarray(x)
    model = l[1]
    criteria = l[2]
    data = l[3].copy()
    fold = l[4]
    results = []
    kfold = KFold(n_splits=fold, shuffle=True)
    kf = kfold.split(data.loc[:, "$<bug"].values)

    for train_index, tune_index in kf:
        train_data = data.ix[train_index]
        tune_data = data.ix[tune_index]
        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        X_test, y_test = tune_data.iloc[:, :-1], tune_data.iloc[:, -1]
        predicted = model(X_train, y_train, X_test, l[0].values())
        res = evaluation(criteria, predicted, y_test, X_test)
        results.append(res)
    try:
        return np.mean(results)
    except:
        return 0.0

