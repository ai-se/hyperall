import multiprocessing as mp
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import os
from time import time
from sklearn.ensemble import RandomForestClassifier

import autosklearn.classification
import pandas as pd
import pickle

def run_default(train, test, perf_measure=None):
    start_time = time()
    # Setting up Training Data
    train_ds = pd.read_csv(train)
    train_columns = [col for col in train_ds.columns if '$' in col]
    train_indep_columns = [col for col in train_columns if '$<' not in col]
    train_dep_columns = [col for col in train_columns if '$<' in col]
    assert (len(train_dep_columns) == 1), "Something is wrong"
    train_dep_column = train_dep_columns[-1]

    train_X = train_ds[train_indep_columns]
    train_Y = [0 if x == 0 else 1 for x in train_ds[train_dep_column]]

    # Setting up Testing Data
    test_ds = pd.read_csv(test)
    test_columns = [col for col in test_ds.columns if '$' in col]
    test_indep_columns = [col for col in test_columns if '$<' not in col]
    test_dep_columns = [col for col in test_columns if '$<' in col]
    assert (len(test_dep_columns) == 1), "Something is wrong"
    test_dep_column = test_dep_columns[-1]

    test_X = test_ds[test_indep_columns]
    test_Y = [0 if x == 0 else 1 for x in test_ds[test_dep_column]]
    assert (test_X.shape[0] == len(test_Y)), "Something is wrong"

    model = RandomForestClassifier()
    model.fit(train_X, train_Y)

    predictions = model.predict(test_X)

    # perf_score = sklearn.metrics.accuracy_score(test_Y, predictions)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_Y, predictions)

    return [confusion_matrix, time()-start_time]


def run_experiment(train, test, perf_measure=None):
    start_time = time()
    # Setting up Training Data
    train_ds = pd.read_csv(train)
    train_columns = [col for col in train_ds.columns if '$' in col]
    train_indep_columns = [col for col in train_columns if '$<' not in col]
    train_dep_columns = [col for col in train_columns if '$<' in col]
    assert(len(train_dep_columns) == 1), "Something is wrong"
    train_dep_column = train_dep_columns[-1]

    train_X = train_ds[train_indep_columns]
    train_Y = [0 if x == 0 else 1 for x in train_ds[train_dep_column]]

    # Setting up Testing Data
    test_ds = pd.read_csv(test)
    test_columns = [col for col in test_ds.columns if '$' in col]
    assert(len(train_columns) == len(test_columns)), "Something is wrong"

    test_indep_columns = [col for col in test_columns if '$<' not in col]
    test_dep_columns = [col for col in test_columns if '$<' in col]
    assert(len(test_indep_columns) + len(test_dep_columns) == len(test_columns)), "Something is wrong"
    assert (len(test_dep_columns) == 1), "Something is wrong"
    test_dep_column = test_dep_columns[-1]

    test_X = test_ds[test_indep_columns]
    test_Y = [0 if x==0 else 1 for x in test_ds[test_dep_column]]
    assert(train_X.shape[0] == len(train_Y)), "Something is wrong"

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='../tmp/autosklearn_cv_example_tmp',
        output_folder='../tmp/autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 3},
        include_estimators=["random_forest", ], exclude_estimators=None,
        include_preprocessors = ["no_preprocessing", ], exclude_preprocessors = None
    )

    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    automl.fit(train_X.copy(), train_Y.copy(), dataset_name='digits')
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(train_X.copy(), train_Y.copy())

    # print(automl.show_models())

    predictions = automl.predict(test_X)
    # perf_score =  sklearn.metrics.accuracy_score(test_Y, predictions)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_Y, predictions)
    return [confusion_matrix, time() - start_time], automl.show_models()


def run_experiment_all(train, test, perf_measure=None):
    start_time = time()
    # Setting up Training Data
    train_ds = pd.read_csv(train)
    train_columns = [col for col in train_ds.columns if '$' in col]
    train_indep_columns = [col for col in train_columns if '$<' not in col]
    train_dep_columns = [col for col in train_columns if '$<' in col]
    assert(len(train_dep_columns) == 1), "Something is wrong"
    train_dep_column = train_dep_columns[-1]

    train_X = train_ds[train_indep_columns]
    train_Y = [0 if x == 0 else 1 for x in train_ds[train_dep_column]]

    # Setting up Testing Data
    test_ds = pd.read_csv(test)
    test_columns = [col for col in test_ds.columns if '$' in col]
    assert(len(train_columns) == len(test_columns)), "Something is wrong"

    test_indep_columns = [col for col in test_columns if '$<' not in col]
    test_dep_columns = [col for col in test_columns if '$<' in col]
    assert(len(test_indep_columns) + len(test_dep_columns) == len(test_columns)), "Something is wrong"
    assert (len(test_dep_columns) == 1), "Something is wrong"
    test_dep_column = test_dep_columns[-1]

    test_X = test_ds[test_indep_columns]
    test_Y = [0 if x==0 else 1 for x in test_ds[test_dep_column]]
    assert(train_X.shape[0] == len(train_Y)), "Something is wrong"

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=220,
        per_run_time_limit=30,
        tmp_folder='../tmp2/autosklearn_cv_example_tmp',
        output_folder='../tmp2/autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 3}
    )

    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    automl.fit(train_X.copy(), train_Y.copy(), dataset_name='digits')
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(train_X.copy(), train_Y.copy())

    # print(automl.show_models())

    predictions = automl.predict(test_X)
    # perf_score =  sklearn.metrics.accuracy_score(test_Y, predictions)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_Y, predictions)
    return [confusion_matrix, time() - start_time], automl.show_models()


if __name__ == '__main__':
    import collections
    Experiment = collections.namedtuple('Experiment', 'train test')
    data_folder = "../Data/DefectPrediction/"
    projects = ['../Data/DefectPrediction/synapse/']

    for project in projects:
        versions = [project + file for file in sorted(os.listdir(project))]
        groups = [Experiment(versions[i-1], versions[i]) for i in range(1, len(versions))]
        results = {}
        for rep in range(20):
            for group_id, group in enumerate(sorted(groups)):
                if group not in results.keys():
                    results[group] = {}
                    results[group]['automl'] = []
                    results[group]['automl_all'] = []
                    results[group]['default'] = []
                    results[group]['automl_model'] = []
                    results[group]['automl_all_model'] = []

                assert (len(group) == 2), "Something is wrong"
                automl, model = run_experiment(group.train, group.test)
                automl_all, model_all = run_experiment_all(group.train, group.test)
                default = run_default(group.train, group.test)
                print(automl, automl_all, default)

                results[group]['automl'].append(automl)
                results[group]['automl_all'].append(automl_all)
                results[group]['default'].append(default)
                results[group]['automl_model'].append(model)
                results[group]['automl_all_model'].append(model_all)

                pickle.dump(results,
                        open('./PickleLocker/' + project.replace(data_folder, '')[:-1] + '_' + str(group_id) + "_" + str(rep) + '.p', 'wb'))
