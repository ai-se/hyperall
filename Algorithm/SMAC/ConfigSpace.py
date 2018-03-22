import pandas as pd
from random import sample
from collections import OrderedDict

class ConfigSpace:
    def __init__(self, filename):
        self.filename = filename
        # Store the content as a dict for faster access
        self.content = OrderedDict()
        self.set_content()

    def set_content(self):
        content = pd.read_csv(self.filename)
        columns = content.columns
        ctrain_indep = [c for c in columns if '$<' not in c]
        ctrain_dep = [c for c in columns if '$<' in c]
        assert (len(ctrain_dep) == 1), "Something is wrong"
        for i in xrange(content.shape[0]):
            row = content.iloc[i]
            indep = row[ctrain_indep].tolist()
            # Single objective
            dep = row[ctrain_dep].tolist()[0]
            if ','.join(map(str, map(float, indep))) in self.content.keys():
                print "Duplicate exists"
                import pdb
                pdb.set_trace()
            self.content[','.join(map(str, map(float, indep)))] = dep

    def get_configurations(self):
        configs = self.content.keys()
        rets = []
        for config in configs:
            rets.append(map(float, config.split(',')))
        assert(len(rets) == len(configs)), "Something is wrong"
        return rets

    def get_performances(self):
        return self.content.values()

    def get_sample_configuration(self, size=1):
        print "get_sample_configuration: ", size
        configs = self.content.keys()
        ret_configs = sample(configs, size)
        ret = [map(float, c.split(',')) for c in ret_configs]
        return ret

    def get_performance(self, config):
        """ Always expects the config to a list of numbers"""
        key = ','.join(map(str, config))
        return self.content[key]

    def get_configuration_size(self):
        return len(self.content.keys())


def _objective_function():
    filename = "~/GIT/hyperall/Config/Data/Apache_AllMeasurements.csv"
    dataset_obj = ConfigSpace(filename)


if __name__ == "__main__":
    _objective_function()