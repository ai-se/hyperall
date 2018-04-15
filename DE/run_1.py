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

    # Setting up Testing Data
    test_ds = pd.concat([pd.read_csv(path) for path in test_paths], ignore_index=True)
    test_columns = [col for col in test_ds.columns if '$' in col]
    test_indep_columns = [col for col in test_columns if '$<' not in col]
    test_dep_columns = [col for col in test_columns if '$<' in col]
    assert (len(test_dep_columns) == 1), "Something is wrong"
    test_dep_column = test_dep_columns[-1]
    test_ds[test_dep_column] = test_ds[test_dep_column].apply(lambda x: 0 if x == 0 else 1)

    return train_ds, test_ds


def call_de(i, x, training_data, testing_data, fold_indexes, goal="Max", term="Early", num_eval=25):
    train_data = training_data.ix[fold_indexes[0]].values
    tune_data = training_data.ix[fold_indexes[1]].values
    test_data = testing_data.values
    num_p = num_eval/5
    de = DE(NP=num_p, GEN=5, Goal=goal, termination=term)
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


def run_DE(train, test, perf_measures, learners, name='', repeats = 1, num_eval=25):
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
    fold = 3
    train_df, test_df = load_data(train, test, name)
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    final_dic={}

    for x in perf_measures:
        temp = {}
        print(x)
        save_pickle_address = 'dump_2/' + str(num_eval) + "_" + x + "_" + train[-1] + '_early.pickle'
        print(save_pickle_address)
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
                                              [train_index, tune_index], "Min", "Late", num_eval=num_eval)
                    else:
                        val, params = call_de(i, x, train_df, test_df,
                                              [train_index, tune_index], "Max", "Early", num_eval=num_eval)
                    l.append(val)
                    l1.append(params)
            total_time = time.time() - start_time
            temp[param_grid[i]['model'].__name__] = [l, l1, total_time]
            print(temp)
        final_dic[x] = temp
        with open(save_pickle_address, 'wb') as handle:
            pickle.dump(final_dic, handle)


if __name__ == '__main__':
    perf_measures = ["f1", "precision"]
    learners = ["svm", "knn", "dt", "rf"]
    data = {"ivy": ['ivy-1.1.csv', 'ivy-1.4.csv', 'ivy-2.0.csv'],
            "lucene": ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
            "poi": ['poi-1.5.csv', 'poi-2.0.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
            "synapse": ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv'],
            "velocity": ['velocity-1.4.csv', 'velocity-1.5.csv'],
            "camel": ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
            "log4j": ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
            "ant": ['ant-1.4.csv', 'ant-1.5.csv', 'ant-1.6.csv'],
            "xerces": ['xerces-1.2.csv', 'xerces-1.3.csv'],
            "jedit": ['jedit-4.1.csv', 'jedit-4.2.csv', 'jedit-4.3.csv']}

    for dataset, datasets in data.items():
        for i in range(len(datasets) - 1):
            run_DE([datasets[i]], [datasets[i + 1]], perf_measures, learners, dataset, repeats=20, num_eval=100)



