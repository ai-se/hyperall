import abc
import logging
from Constant import MAXINT
import numpy as np


class AcquisitionFunctionMaximizer(object):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method ``_maximize`` must be implemented.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(self, acquisition_function, config_space=None):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.acquisition_function = acquisition_function  # LogEI
        self.config_space = config_space
        self.rng = np.random.RandomState(seed=1)

    def maximize(self, runhistory, num_points):
        """Maximize acquisition function using ``_maximize``."""
        return [t[1] for t in self._maximize(runhistory, num_points)]

    def _maximize(self, runhistory, num_points):
        """Implements acquisition function maximization."""
        raise NotImplementedError()

    def _sort_configs_by_acq_value(self, configs):
        """Sort the given configurations by acquisition value"""
        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]

    def get_one_exchange_neighbourhood(self, star_config):
        def diff(config_a, config_b):
            assert(len(config_a) == len(config_b)), "Something is wrong"
            count = 0
            for c_a, c_b in zip(config_a, config_b):
                if c_a != config_b: count += 1
            if count == 1: return True
            else: return False
        ret = []
        # Find all the configuration in the config space which is different by just one element
        for config in self.config_space.get_configurations():
            if diff(star_config, config) is True: ret.append(config)
        return ret



class LocalSearch(AcquisitionFunctionMaximizer):
    """Implementation of SMAC's local search.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional

    epsilon: float
        In order to perform a local move one of the incumbent's neighbors
        needs at least an improvement higher than epsilon
    max_iterations: int
        Maximum number of iterations that the local search will perform

    """

    def __init__(self, acquisition_function, config_space, rng=None, epsilon=0.00001, max_iterations=None):
        super(LocalSearch, self).__init__(acquisition_function, config_space)
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def _maximize(self, runhistory, num_points, *args):
        """Starts a local search from the given start point and quits if either the max number of steps is reached or no 
        neighbor with an higher improvement was found."""
        # Vivek: Number of configurations sampled by local search is min(number of configurations sampled, 10)
        num_configurations_by_local_search = self._calculate_num_points(num_points, runhistory)
        # Vivek: Initiate local search with best configurations from previous runs
        init_points = self._get_initial_points(num_configurations_by_local_search, runhistory)
        configs_acq = []

        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, configuration = self._one_iter(start_point)
            configs_acq.append((acq_val, configuration))

        # shuffle for random tie-break
        self.rng.shuffle(configs_acq)

        # sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])

        return configs_acq

    def _calculate_num_points(self, num_points, runhistory):
        num_configurations_by_local_search = min(len(runhistory.get_all_configs()), num_points)
        return num_configurations_by_local_search

    def _get_initial_points(self, num_configurations_by_local_search, runhistory):
        if runhistory.empty():
            init_points = self.config_space.sample_configuration(size=num_configurations_by_local_search)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = runhistory.get_all_configs()
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(configs_previous_runs)
            num_configs_local_search = min(len(configs_previous_runs_sorted), num_configurations_by_local_search)
            init_points = list(map(lambda x: x[1], configs_previous_runs_sorted[:num_configs_local_search]))
        return init_points

    def _one_iter(self, start_point, *args):
        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent], *args)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:
            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                self.logger.warning(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

            # Get neighborhood of the current incumbent by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns a list.
            all_neighbors = self.get_one_exchange_neighbourhood(incumbent)

            for neighbor in all_neighbors:
                acq_val = self.acquisition_function([neighbor], *args)
                neighbors_looked_at += 1

                if acq_val > acq_val_incumbent + self.epsilon:
                    self.logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or (self.max_iterations is not None and local_search_steps == self.max_iterations):
                self.logger.debug("Local search took %d steps and looked at %d "
                                  "configurations. Computing the acquisition "
                                  "value for one configuration took %f seconds"
                                  " on average.",
                                  local_search_steps, neighbors_looked_at,
                                  np.mean(time_n))
                break

        return acq_val_incumbent, incumbent


class RandomSearch(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(self, runhistory, num_points, _sorted=False, *args):
        rand_configs = self.config_space.get_sample_configuration(size=num_points)

        if _sorted:
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]


class InterleavedLocalAndRandomSearch(AcquisitionFunctionMaximizer):
    """Implements SMAC's default acquisition function optimization."""

    def __init__(self, acquisition_function, config_space, rng=None,):
        super(InterleavedLocalAndRandomSearch, self).__init__(acquisition_function, config_space)
        self.random_search = RandomSearch(acquisition_function, config_space)
        self.local_search = LocalSearch(acquisition_function, config_space)

    def maximize(self, runhistory, num_points, *args):
        next_configs_by_local_search = self.local_search._maximize(runhistory, 10)
        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(runhistory,
                                                    num_points - len(next_configs_by_local_search), _sorted=True)

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (next_configs_by_random_search_sorted + next_configs_by_local_search)
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value, self.config_space)
        return challengers

    def _maximize(self, runhistory, num_points):
        raise NotImplementedError()


class ChallengerList(object):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    challengers : list
        List of challengers (without interleaved random configurations)

    configuration_space : ConfigurationSpace
        ConfigurationSpace from which to sample new random configurations.
    """

    def __init__(self, challengers, configuration_space):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._next_is_random = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers) and not self._next_is_random:
            raise StopIteration
        elif self._next_is_random:
            self._next_is_random = False
            config = self.configuration_space.sample_configuration()
            return config
        else:
            self._next_is_random = True
            config = self.challengers[self._index]
            self._index += 1
            return config
