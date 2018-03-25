import sys

sys.dont_write_bytecode = True
import os
from collections import OrderedDict
from learner import *
from DE import DE
from random import seed
import time
import pandas as pd
import numpy as np

## Global Bounds
cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
data = {"@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "@lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "@poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "@synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "@velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "@camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "@jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "@log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "@xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "@xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }
learners_para_dic = OrderedDict([("max_depth", 1), ("medianTop", 1)])
learners_para_bounds=[(3, 4), (0, 1)]
learners_para_categories=["categorical", "categorical"]
learners=[DT, RF, SVM, NB, KNN, LR]
measures=["Dist2Heaven", "LOC_AUC"]
repeats=10


def run_DE(train, test, perf_measures, learners):
    """
     This function would take a train dataset and would find parameter using DE by performing CV with fold == 5.
    :param train: file path of train
    :param test: file path of test
    :param learner:
    :param perf_measure: Accuracy, recall etc.
    :return: performance measure
    """
    def call_de(i, x, train_data, test_data, goal="Max", term="Early"):
        de = DE(GEN=2, Goal=goal, termination=term)
        v, pareto = de.solve(process, OrderedDict(param_grid[i]['learners_para_dic']),
                             param_grid[i]['learners_para_bounds'], param_grid[i]['learners_para_categories'],
                             param_grid[i]['model'], x, train_data)
        params = v.ind
        predicted_tune = param_grid[i]['model'](train_data[:, :-1], train_data[:, -1],
                                                test_data[:, :-1], params)
        predicted_default = param_grid[i]['model'](train_data[:, :-1], train_data[:, -1],
                                                   test_data[:, :-1], None)
        val_tune = evaluate(x, predicted_tune, test_data[:, -1])
        val_predicted = evaluate(x, predicted_default, test_data[:, -1])
        print("For measure %s: default=%s, predicted=%s" % (x, val_predicted, val_tune))
        return val_tune, params

    seed(1)
    np.random.seed(1)
    paths = [os.path.join(data_path, file_name) for file_name in data[res]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    ### getting rid of first 3 columns
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)

    final_dic={}
    temp={}
    for x in perf_measures:
        temp = {}
        print(x)
        for i in learners:
            naive_clf = learner(i, False)
            experiment([train_data[:, :-1], test_data[:, :-1]],
                       [train_data[:, -1], test_data[:, -1]], naive_clf, i, split_bool=True)
            l = []
            l1 = []
            start_time = time.time()
            print("Learner: %s" % i)
            for r in xrange(repeats):
                train_data = data_bins[r]
                print("Repeating: %s" % r)
                if x == "d2h":
                    val, params = call_de(i, x, train_data, test_data, "Min", "Late")
                else:
                    val, params = call_de(i, x, train_data, test_data, "Max", "Late")
                l.append(val)
                l1.append(params)
            total_time = time.time() - start_time
            temp[param_grid[i]['model'].__name__] = [l, l1, total_time]
            print(temp)
        final_dic[x] = temp
    with open('../dump/' + '_early.pickle', 'wb') as handle:
        pickle.dump(final_dic, handle)


if __name__ == '__main__':
    de_fft('@ivy')

