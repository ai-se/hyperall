import pandas as pd
import os


class ObjetiveFunction:
    def __init__(self, filename):
        self.filename = filename
        # Store the content as a dict for faster access
        self.content = {}
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
            if ','.join(map(str, indep)) in self.content.keys():
                print "Duplicate exists"
                import pdb
                pdb.set_trace()
            self.content[','.join(map(str, indep))] = dep


def _objective_function():
    filename = "~/GIT/hyperall/Config/Data/Apache_AllMeasurements.csv"
    dataset_obj = ObjetiveFunction(filename)


if __name__ == "__main__":
    _objective_function()