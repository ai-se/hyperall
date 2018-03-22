import pandas as pd
import os
from smbo import smbo
from RandomForestWithInstances import RandomForestWithInstances
from ConfigSpace import ConfigSpace
from InitialDesign import RandomInitialDesign
from Intensification import Intensifier
from RunHistory import RunHistory
from AcquisitionFunctionOptimizer import InterleavedLocalAndRandomSearch
from AcquisitionFunction import LogEI


def SMAC(configuration_space, budget=50):
    model = RandomForestWithInstances()
    # TODO: Check if the indep and dep values need to be changed PCA etc.

    initial_design = RandomInitialDesign(config_space=configuration_space)
    runhistory = RunHistory()
    acquition_func = LogEI(model=model)
    acq_optimizer = InterleavedLocalAndRandomSearch(config_space=configuration_space, acquisition_function=acquition_func)
    intensifier = Intensifier
    solver = smbo(configuration_space, initial_design, intensifier, model, runhistory, acq_optimizer, acquition_func, budget)
    incumbent = solver.run()
    return incumbent

if __name__ == "__main__":
    filename = "/Users/viveknair/GIT/hyperall/Config/Data/Apache_AllMeasurements.csv"
    config_space = ConfigSpace(filename=filename)
    SMAC(configuration_space=config_space)