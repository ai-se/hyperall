import numpy as np
from RunHistory import RunKey

def get_random_generator(number=None):
    if number is None:
        return np.random.RandomState(1)
    else:
        return np.random.RandomState(number)


def sum_cost(config, run_history, instance_seed_pairs=None):
    """Return the sum of costs of a configuration.
    This is the sum of costs of all instance-seed pairs.
    Parameters
    ----------
    config : Configuration
        Configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        List of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.
    Returns
    ----------
    sum_cost: float
        Sum of costs of config
    """
    return np.sum(_cost(config, run_history, instance_seed_pairs))


def _cost(config, run_history, instance_seed_pairs=None):
    """Return array of all costs for the given config for further calculations.
    Parameters
    ----------
    config : Configuration
        Configuration to calculate objective for
    run_history : RunHistory
        RunHistory object from which the objective value is computed.
    instance_seed_pairs : list, optional (default=None)
        List of tuples of instance-seeds pairs. If None, the run_history is
        queried for all runs of the given configuration.
    Returns
    -------
    Costs: list
        Array of all costs
    """
    try:
        id_ = run_history.config_ids[config]
    except KeyError:  # challenger was not running so far
        return []

    if instance_seed_pairs is None:
        instance_seed_pairs = run_history.get_runs_for_config(config)

    costs = []
    for i, r in instance_seed_pairs:
        k = RunKey(id_, i, r)
        costs.append(run_history.data[k].cost)
    return costs