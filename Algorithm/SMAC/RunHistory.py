
class RunHistoryElement:
    def __init__(self, gen_id, performance):
        self.gen_id = gen_id  # Generation or iteration where it was sampled
        self.performance = performance


class RunHistory:
    def __init__(self):
        self.runhistory = {}

    def add(self, gen_id, config, performance=None):
        key = ','.join(map(str, config))
        if key in self.runhistory.keys():
            print "This has been sampled and measured in a previous iteration. Please check the code"
            exit()
        self.runhistory[key] = RunHistoryElement(gen_id, performance)


    def get_all_configs(self):
        """Return all configurations in this RunHistory object"""
        return self.runhistory.keys()