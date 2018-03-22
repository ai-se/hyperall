from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RandomForestWithInstances:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=10, bootstrap=True, min_samples_split=3, min_samples_leaf=3, max_depth=20,
                          max_leaf_nodes= 2**20, random_state=42, )

    def train(self, independents, dependents):
        assert (len(independents) == len(dependents)), "Something is wrong"
        np_independents = np.array(independents)
        np_dependents = np.array(dependents)
        self.rf.fit(np_independents, np_dependents)

    def predict(self, independents):
        means = []
        vars = []
        np_independents = np.array(independents)

        for id in range(np_independents.shape[0]):
            preds = []
            for pred in self.rf.estimators_:
                preds.append(pred.predict(np_independents[id].reshape(1, -1))[0])
            means.append(np.mean(preds))
            vars.append(np.var(preds))
        return np.array(means).reshape(-1, 1), np.array(vars).reshape(-1, 1)