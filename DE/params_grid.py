from collections import OrderedDict
from learner import DTC, RF, SVClassifier, KNN

param_grid = {}
# Decision Trees
param_grid['dt'] = {
    "model": RF,
    "learners_para_dic": OrderedDict([("max_features", 1), ("min_samples_split", 1),
                                      ("min_samples_leaf", 1), ("max_depth", 1)]),
    "learners_para_bounds": [(0.1, 1.0), (2, 30), (1, 21), (1, 21)],
    "learners_para_categories": ["continuous", "integer", "integer", "integer"]
}


# Random Forests
param_grid['rf'] = {
    "model": RF,
    "learners_para_dic": OrderedDict([("max_features", 1), ("min_samples_split", 1),
                                      ("min_samples_leaf", 1), ("n_estimators", 1)]),
    "learners_para_bounds": [(0.1, 1.0), (2, 30), (1, 21), (10, 150)],
    "learners_para_categories": ["continuous", "integer", "integer", "integer"]
}

# Support Vector Machine
param_grid['svm'] = {
    "model": SVRegressor,
    "learners_para_dic": OrderedDict([("C", 1), ("kernel", 1), ("coef0", 1), ("gamma", 1)]),
    "learners_para_bounds": [(1, 100), ('rbf', 'linear', 'sigmoid', 'poly'), (0.1, 1.0), (0.1, 1.0)],
    "learners_para_categories": ["integer", "categorical", "continuous", "continuous"]
}

# KNN
param_grid['knn'] = {
    "model": KNN,
    "learners_para_dic": OrderedDict([("n_neighbors", 1), ("weights", 1)]),
    "learners_para_bounds": [(2, 10), ('uniform', 'distance')],
    "learners_para_categories": ["integer", "categorical"]
}

