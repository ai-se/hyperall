import collections
RunKey = collections.namedtuple('RunKey', ['config_id', 'instance_id', 'seed'])


class RunHistory:
    def __init__(self):
        self.runhistory = collections.OrderedDict()

    def add(self, config, performance):
        key = ','.join(map(str, config))
        if key in self.runhistory.keys():
            print "This has been sampled and measured in a previous iteration. Please check the code"
            import pdb
            pdb.set_trace()
        self.runhistory[key] = performance

    def get_all_configs(self):
        """Return all configurations in this RunHistory object"""
        configs = [map(float, x.split(',')) for x in self.runhistory.keys()]
        return configs

    def get_candidates(self):
        configs = [map(float, x.split(',')) for x in self.runhistory.keys()]
        return configs, self.runhistory.values()

    def empty(self):
        if len(self.runhistory.keys()) == 0: return True
        else: return False

    def get_incumbent(self, lessismore=True):
        if lessismore is True:
            return min(self.runhistory.values())
        else:
            return max(self.runhistory.values())

    def is_present(self, config):
        t_config = ','.join(map(str, map(float, config)))
        if t_config in self.runhistory.keys(): return True
        else: return False
