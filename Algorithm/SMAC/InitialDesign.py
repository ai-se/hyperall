from random import sample

class InitialDesign:
    def __init__(self, config_space):
        self.config_space = config_space
        # TODO: the input should be ConfigSpace not a list or dictionary
    def run(self, size):
        raise NotImplementedError


class RandomInitialDesign(InitialDesign):
    def run(self, size=1):
        # Returns a list of configurations.
        return self.config_space.get_sample_configuration(size)