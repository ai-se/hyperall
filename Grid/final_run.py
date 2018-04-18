import sys

sys.dont_write_bytecode = True
import os
from collections import OrderedDict
from learner import *
from Grid_1 import Grid
import time
import pandas as pd
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


def call_grid(i, x, training_data, testing_data, fold, goal="Max", eval_time=3600, lifes=5):
    # test_data = testing_data.values
    grid_optimizer = Grid(NP=10, Goal=goal, eval_time=eval_time, num_lifes=lifes)
    v, num_evals, tuning_time = grid_optimizer.solve(process, OrderedDict(param_grid[i]['learners_para_dic']),
                                         param_grid[i]['learners_para_bounds'],
                                         param_grid[i]['learners_para_categories'],
                                         param_grid[i]['model'], x, training_data, fold)
    params = v.ind.values()
    start_time = time.time()
    predicted_tune = param_grid[i]['model'](training_data.iloc[:, :-1], training_data.iloc[:, -1],
                                            testing_data.iloc[:, :-1], params)
    predicted_default = param_grid[i]['model'](training_data.iloc[:, :-1], training_data.iloc[:, -1],
                                               testing_data.iloc[:, :-1], None)
    val_tune = evaluation(x, predicted_tune, testing_data.iloc[:, -1], testing_data.iloc[:, :-1])
    val_predicted = evaluation(x, predicted_default, testing_data.iloc[:, -1], testing_data.iloc[:, :-1])
    print("For measure %s: default=%s, predicted=%s" % (x, val_predicted, val_tune))
    tuning_time += time.time() - start_time
    # print(val_tune, params, num_evals, tuning_time)
    return val_tune, params, num_evals, tuning_time


def run_Grid(train, test, perf_measures, learners, name='', repeats = 1, eval_time=3600, lifes=5):
    """
     This function would take a train dataset and would find parameter using DE by performing CV with fold == 5.
    :param train: file path of train
    :param test: file path of test
    :param learner:
    :param perf_measure: Accuracy, recall etc.
    :return: performance measure
    """
    fold = 3
    train_df, test_df = load_data(train, test, name)
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    final_dic = {}

    for x in perf_measures:
        temp = {}
        save_pickle_address = 'dump_2/%s_grid_results/' % x
        save_pickle_address += x + "_" + test[-1] + '_early.pickle'
        print(save_pickle_address)
        for i in learners:
            l, l1, l2, l3 = [], [], [], []
            print("Learner: %s" % i)
            for r in xrange(repeats):
                print("Repeating: %s" % r)
                if x == "d2h":
                    val, params, num_evals, tuning_time = call_grid(i, x, train_df, test_df, fold, "Min", eval_time=eval_time,
                                                                  lifes=lifes)
                else:
                    val, params, num_evals, tuning_time = call_grid(i, x, train_df, test_df, fold, "Max", eval_time=eval_time,
                                                                  lifes=lifes)
                l.append(val)
                l1.append(params)
                l2.append(num_evals)
                l3.append(tuning_time)
            temp[param_grid[i]['model'].__name__] = [l, l1, l2, l3]
            print(param_grid[i]['model'].__name__, l2, l3)
            # print(temp)
        final_dic[x] = temp
        with open(save_pickle_address, 'wb') as handle:
            pickle.dump(final_dic, handle)


'''
if __name__ == '__main__':
    perf_measures = ["precision", "f1"]
    learners = ["svm", "knn", "dt", "rf"]
    data = ["camel"]

    for dataset in data:
        datasets = [x[2] for x in os.walk("../Data/DefectPrediction/%s" % dataset)][0]
        #print(datasets)
        run_DE([datasets[0]], [datasets[0 + 1]], perf_measures, learners, dataset, repeats=20, num_eval=50)
        #for i in range(2, len(datasets) - 1):
        #    run_DE([datasets[i]], [datasets[i+1]], perf_measures, learners, dataset, repeats=20, num_eval=50)
'''

if __name__ == '__main__':
    perf_measures = ["f1", "precision"]
    learners = ["knn", "svm", "dt", "rf"]
    data = {"ant": ["ant-1.4.csv", "ant-1.5.csv", "ant-1.6.csv", "ant-1.7.csv"],
            "ivy": ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],
            "lucene": ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],
            "poi": ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],
            "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],
            "velocity": ["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"],
            "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"],
            "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"],
            "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"],
            "xerces": ["xerces-1.1.csv", "xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
            }
    data = {
            "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"],
            "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"],
            "xerces": ["xerces-1.1.csv", "xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
            }
    for dataset, datasets in data.items():
        for i in range(len(datasets) - 1):
            run_Grid([datasets[i]], [datasets[i + 1]], perf_measures,
                   learners, dataset, repeats=10, eval_time=3600, lifes=5)


