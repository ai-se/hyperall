import sys

sys.dont_write_bytecode = True
import os
from collections import OrderedDict
from sklearn.model_selection import GroupKFold, KFold
from learner import *
from DE import DE
from random import seed
import time
import pandas as pd
import numpy as np
from params_grid import param_grid

## Global Bounds
cwd = os.getcwd()
data_path = os.path.join(cwd, "../Data/DefectPrediction/")
print(data_path)
data = ["ivy", "lucene", "poi", "synapse", "velocity",
        "camel", "log4j", "ant", "xerces", "jedit"]


def load_data(train, test, name):
    # Setting up Training Data
    train_paths = [os.path.join(data_path, name, file_name) for file_name in train]
    test_paths = [os.path.join(data_path, name, file_name) for file_name in test]
    #print(train_paths, test_paths)
    train_ds = pd.concat([pd.read_csv(path) for path in train_paths], ignore_index=True)
    train_columns = [col for col in train_ds.columns if '$' in col]
    train_indep_columns = [col for col in train_columns if '$<' not in col]
    train_dep_columns = [col for col in train_columns if '$<' in col]
    assert (len(train_dep_columns) == 1), "Something is wrong"
    train_dep_column = train_dep_columns[-1]
    train_ds[train_dep_column] = train_ds[train_dep_column].apply(lambda x: 0 if x == 0 else 1)

    #train_X = train_ds[train_indep_columns]
    #train_Y = [0 if x == 0 else 1 for x in train_ds[train_dep_column]]

    # Setting up Testing Data
    test_ds = pd.concat([pd.read_csv(path) for path in test_paths], ignore_index=True)
    test_columns = [col for col in test_ds.columns if '$' in col]
    test_indep_columns = [col for col in test_columns if '$<' not in col]
    test_dep_columns = [col for col in test_columns if '$<' in col]
    assert (len(test_dep_columns) == 1), "Something is wrong"
    test_dep_column = test_dep_columns[-1]
    test_ds[test_dep_column] = test_ds[test_dep_column].apply(lambda x: 0 if x == 0 else 1)

    #test_X = test_ds[test_indep_columns]
    #test_Y = [0 if x == 0 else 1 for x in test_ds[test_dep_column]]
    #assert (test_X.shape[0] == len(test_Y)), "Something is wrong"
    #return [train_X.values, np.array(train_Y)], [test_X.values, np.array(test_Y)]
    return train_ds, test_ds


def call_de(i, x, training_data, testing_data, fold_indexes, goal="Max", term="Early"):
    train_data = training_data.ix[fold_indexes[0]].values
    tune_data = training_data.ix[fold_indexes[1]].values
    test_data = testing_data.values
    de = DE(NP=5, GEN=5, Goal=goal, termination=term)
    v, pareto = de.solve(process, OrderedDict(param_grid[i]['learners_para_dic']),
                         param_grid[i]['learners_para_bounds'], param_grid[i]['learners_para_categories'],
                         param_grid[i]['model'], x, [train_data, tune_data])
    params = v.ind.values()
    training_data = training_data.values
    predicted_tune = param_grid[i]['model'](training_data[:, :-1], training_data[:, -1],
                                            test_data[:, :-1], params)
    predicted_default = param_grid[i]['model'](training_data[:, :-1], training_data[:, -1],
                                               test_data[:, :-1], None)
    val_tune = evaluation(x, predicted_tune, test_data[:, -1], test_data[:, :-1])
    val_predicted = evaluation(x, predicted_default, test_data[:, -1], test_data[:, :-1])
    print("For measure %s: default=%s, predicted=%s" % (x, val_predicted, val_tune))
    return val_tune, params


def run_DE(train, test, perf_measures, learners, name=''):
    """
     This function would take a train dataset and would find parameter using DE by performing CV with fold == 5.
    :param train: file path of train
    :param test: file path of test
    :param learner:
    :param perf_measure: Accuracy, recall etc.
    :return: performance measure
    """
    seed(47)
    np.random.seed(47)
    repeats = 20
    fold = 3
    train_df, test_df = load_data(train, test, name)
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    final_dic={}
    save_pickle_address = 'dump/' + "25_" + perf_measures[0] + "_" + train[-1] + '_early.pickle'
    print(save_pickle_address)

    for x in perf_measures:
        temp = {}
        print(x)
        for i in learners:
            l = []
            l1 = []
            start_time = time.time()
            print("Learner: %s" % i)
            for r in xrange(repeats):
                print("Repeating: %s" % r)
                kfold = KFold(n_splits=fold, shuffle=True)
                kf = kfold.split(train_df.loc[:, "$<bug"].values)
                for train_index, tune_index in kf:
                    if x == "d2h":
                        val, params = call_de(i, x, train_df, test_df,
                                              [train_index, tune_index], "Min", "Late")
                    else:
                        val, params = call_de(i, x, train_df, test_df,
                                              [train_index, tune_index], "Max", "Early")
                    l.append(val)
                    l1.append(params)
            total_time = time.time() - start_time
            temp[param_grid[i]['model'].__name__] = [l, l1, total_time]
            print(temp)
        final_dic[x] = temp
    with open(save_pickle_address, 'wb') as handle:
        pickle.dump(final_dic, handle)


if __name__ == '__main__':
    perf_measures = ["f1"]
    learners = ["svm", "knn", "dt", "rf"]
    for dataset in data:
        datasets = [x[2] for x in os.walk("../Data/DefectPrediction/%s" % dataset)][0]
        for i in range(len(datasets)-1):
            run_DE([datasets[i]], [datasets[i+1]], perf_measures, learners, dataset)



