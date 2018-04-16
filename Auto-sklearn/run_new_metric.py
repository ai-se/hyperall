# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics

from sklearn.metrics import auc


def accuracy_wk(raw_solution, raw_prediction, dummy):
    def subtotal(x):
        xx = [0]
        for i, t in enumerate(x):
            xx += [xx[-1] + t]
        return xx[1:]

    def get_recall(true):
        total_true = float(len([i for i in true if i == 1]))
        hit = 0.0
        recall = []
        for i in range(len(true)):
            if true[i] == 1:
                hit += 1
            recall += [hit / total_true if total_true else 0.0]
        return recall

    solution = [x%100 for x in raw_solution]
    loc = [int(x/100) for x in raw_solution]
    prediction = [x%10 for x in raw_prediction]
    df1 = pd.DataFrame(prediction, columns=["prediction"])
    df2 = pd.DataFrame(solution, columns=["$<bug"])
    df3 = pd.DataFrame(loc, columns=["$loc"])
    data = pd.concat([df3, df2, df1], axis=1)

    print(data)

    print(prediction, len(prediction), data.shape[0])
    # print(data)

    data.sort_values(by=["$<bug", "$loc"], ascending=[0, 1], inplace=True)
    x_sum = float(sum(data['$loc']))
    x = data['$loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)

    # get  AUC_optimal
    yy = get_recall(data['$<bug'].values)
    xxx = [i for i in xx if i <= 0.2]
    yyy = yy[:len(xxx)]
    s_opt = round(auc(xxx, yyy), 3)

    # get AUC_worst
    xx = subtotal(x[::-1])
    yy = get_recall(data['$<bug'][::-1].values)
    xxx = [i for i in xx if i <= 0.2]
    yyy = yy[:len(xxx)]
    try:
        s_wst = round(auc(xxx, yyy), 3)
    except:
        # print "s_wst forced = 0"
        s_wst = 0

    # get AUC_prediction
    data.sort_values(by=["prediction", "$loc"], ascending=[0, 1], inplace=True)
    x = data['$loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)
    yy = get_recall(data['$<bug'].values)
    xxx = [k for k in xx if k <= 0.2]
    yyy = yy[:len(xxx)]
    try:
        s_m = round(auc(xxx, yyy), 3)
    except:
        return 0

    Popt = (s_m - s_wst) / (s_opt - s_wst)
    return round(Popt, 2)


def main(train, test):
    train_ds = pd.read_csv(train)
    train_columns = [col for col in train_ds.columns if '$' in col]
    train_indep_columns = [col for col in train_columns if '$<' not in col]
    train_dep_columns = [col for col in train_columns if '$<' in col]
    assert (len(train_dep_columns) == 1), "Something is wrong"
    train_dep_column = train_dep_columns[-1]

    mask = 100
    train_X = train_ds[train_indep_columns]
    temp_train_Y = [0 if x == 0 else 1 for x in train_ds[train_dep_column]]
    train_Y = [x+y for x,y in zip(train_X['$loc'].apply(lambda x:x*mask).tolist(), temp_train_Y)]

    # Setting up Testing Data
    test_ds = pd.read_csv(test)
    test_columns = [col for col in test_ds.columns if '$' in col]
    assert (len(train_columns) == len(test_columns)), "Something is wrong"

    test_indep_columns = [col for col in test_columns if '$<' not in col]
    test_dep_columns = [col for col in test_columns if '$<' in col]
    assert (len(test_indep_columns) + len(test_dep_columns) == len(test_columns)), "Something is wrong"
    assert (len(test_dep_columns) == 1), "Something is wrong"
    test_dep_column = test_dep_columns[-1]

    test_X = test_ds[test_indep_columns]
    temp_test_Y = [0 if x == 0 else 1 for x in test_ds[test_dep_column]]
    test_Y = [x + y for x, y in zip(test_X['$loc'].apply(lambda x: x * mask).tolist(), temp_test_Y)]
    assert (test_X.shape[0] == len(test_Y)), "Something is wrong"

    import pdb
    pdb.set_trace()

    # X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Third example: Use own accuracy metric with additional argument
    print("#"*80)
    print("Use self defined accuracy with additional argument")
    accuracy_scorer = autosklearn.metrics.make_scorer(
        name="popt_add",
        score_func=accuracy_wk,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
        dummy=train_ds,
    )
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        seed=1,
    )
    cls.fit(train_X, train_Y, metric=accuracy_scorer)

    predictions = cls.predict(test_X)
    print("popt_20 score {:g}".format(accuracy_wk(0, predictions, test_ds)))


if __name__ == "__main__":
    main('../Data/DefectPrediction/ant/ant-1.3.csv', '../Data/DefectPrediction/ant/ant-1.4.csv')