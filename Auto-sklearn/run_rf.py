
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


def run_experiment(train, test, seed, run_count, perf_measure):
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
        time_left_for_this_task=3600,
        per_run_time_limit=30,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 3},
        include_estimators=["random_forest", ], exclude_estimators=None,
        include_preprocessors = ["no_preprocessing", ], exclude_preprocessors = None,
        ensemble_size=0,
        seed=seed,
    )

    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    automl.fit(train_X.copy(), train_Y.copy(), metric=perf_measure) #autosklearn.metrics.f1)


    # print(automl.show_models())
    automl.fit_ensemble(train_Y, ensemble_size=50)

    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(train_X.copy(), train_Y.copy())
    predictions = automl.predict(test_X)

    # perf_score =  sklearn.metrics.accuracy_score(test_Y, predictions)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_Y, predictions)
    return [confusion_matrix, time() - start_time], automl.show_models()




if __name__ == '__main__':
    import collections
    Experiment = collections.namedtuple('Experiment', 'train test')
    data_folder = "../Data/DefectPrediction/"

    projects = [
        # '../Data/DefectPrediction/ant/',
        # '../Data/DefectPrediction/camel/',
        # '../Data/DefectPrediction/ivy/',
        # '../Data/DefectPrediction/jedit/',
        #  '../Data/DefectPrediction/log4j/',
        # '../Data/DefectPrediction/lucene/',
        # '../Data/DefectPrediction/poi/',
        # '../Data/DefectPrediction/synapse/',
        # '../Data/DefectPrediction/velocity/',
        # '../Data/DefectPrediction/xerces/'
    ]
    perf_measures = {
                        "Prec": autosklearn.metrics.f1,
                        "F1": autosklearn.metrics.precision
                     }
    for perf_measure in perf_measures.keys():
        for project in projects:
            versions = [project + file for file in sorted(os.listdir(project))]
            groups = [Experiment(versions[i-1], versions[i]) for i in range(1, len(versions))]
            results = {}
            for rep in range(10):
                for group in groups:
                    if group not in results.keys():
                        results[group] = {}
                        results[group]['automl'] = []

                    assert(len(group) == 2), "Something is wrong"
                    default = run_default(group.train, group.test)
                    automl, model = run_experiment(group.train, group.test, run_count=eval, seed=rep, perf_measure=perf_measures[perf_measure])

                    results[group]['automl'].append(automl)

                pickle.dump(results, open('./PickleLocker_rf_/' + perf_measure + '_' + str(eval)+'/' + project.replace(data_folder, '')[:-1] + '_' + str(rep) + '.p', 'wb'))

