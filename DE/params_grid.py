from collections import OrderedDict
from learner import DTC, RF, SVM, KNN

param_grid = {}
# Decision Trees
param_grid['dt'] = {
    "model": DTC,
    "learners_para_dic": OrderedDict([("criterion", 1), ("max_features", 1), ("min_samples_split", 1),
                                      ("min_samples_leaf", 1), ("max_depth", 1)]),
    "learners_para_bounds": [("gini", "entropy"), (0.1, 1.0), (2, 30), (1, 21), (1, 21)],
    "learners_para_categories": ["categorical", "continuous", "integer", "integer", "integer"]
}


# Random Forests
param_grid['rf'] = {
    "model": RF,
    "learners_para_dic": OrderedDict([("criterion", 1), ("max_features", 1), ("min_samples_split", 1),
                                      ("min_samples_leaf", 1), ("n_estimators", 1)]),
    "learners_para_bounds": [("gini", "entropy"), (0.1, 1.0), (2, 30), (1, 21), (10, 100)],
    "learners_para_categories": ["categorical", "continuous", "integer", "integer", "integer"]
}

# Support Vector Machine
param_grid['svm'] = {
    "model": SVM,
    "learners_para_dic": OrderedDict([("C", 1), ("kernel", 1), ("coef0", 1), ("gamma", 1)]),
    "learners_para_bounds": [(1, 100), ('rbf', 'sigmoid'), (0.1, 1.0), (0.1, 1.0)],
    "learners_para_categories": ["integer", "categorical", "continuous", "continuous"]
}

# KNN
param_grid['knn'] = {
    "model": KNN,
    "learners_para_dic": OrderedDict([("n_neighbors", 1), ("weights", 1)]),
    "learners_para_bounds": [(2, 10), ('uniform', 'distance')],
    "learners_para_categories": ["integer", "categorical"]
}

