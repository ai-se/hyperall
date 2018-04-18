from __future__ import print_function, division

import sys
sys.dont_write_bytecode = True

from collections import OrderedDict, namedtuple
from random import random, randint, uniform, seed, choice, sample
import numpy as np
from random import shuffle
from sklearn.model_selection import ParameterGrid
import math
import time

Individual = namedtuple('Individual', 'ind fit')

class Grid(object):
    def __init__(self, NP=10, Goal="Max", num_lifes=5, eval_time=600):
        self.NP=NP
        self.GOAL=Goal
        self.eval_time = eval_time
        self.termination_lifes = num_lifes
        self.start_time = 0
        self.num_evals = 0
        self.params = {}
        self.grids = []
        #seed(1)
        #np.random.seed(1)

    def grids_making(self):
        grids = ParameterGrid(self.params)
        self.grids = [grid for grid in grids]
        shuffle(self.grids)


    def pop(self):
        l = []
        grid_index_start = self.num_evals * 10
        if self.para_len == 2:
            if self.num_evals == 1:
                for i in range(grid_index_start, len(self.grids)):
                    dic = OrderedDict()
                    for j in self.para_dic.keys():
                        dic[j] = self.grids[i][j]
                    l.append(dic)
            elif self.num_evals == 0:
                for i in range(self.NP):
                    dic = OrderedDict()
                    for j in self.para_dic.keys():
                        dic[j] = self.grids[grid_index_start + i][j]
                    l.append(dic)
            else:
                l = []
        else:
            for i in range(self.NP):
                dic = OrderedDict()
                for j in self.para_dic.keys():
                    dic[j] = self.grids[grid_index_start + i][j]
                l.append(dic)
        if l:
            self.num_evals += 1
        return l


    def pop_1(self):
        l = []
        if self.para_len > 2:
            for i in range(self.NP):
                dic = OrderedDict()
                for j in self.para_dic.keys():
                    dic[j] = self.grids[i][j]
                l.append(dic)
        else:
            for grid in self.grids:
                dic = OrderedDict()
                for j in self.para_dic.keys():
                    dic[j] = grid[j]
                l.append(dic)
        return l


    ## Need a tuple for integer and continuous variable but need the whole list for category
    def search_space(self):
        if self.para_len == 2:
            self.params[self.para_dic.keys()[0]] = range(self.bounds[0][0], self.bounds[0][1]+1)
            self.params[self.para_dic.keys()[1]] = [self.bounds[1][0], self.bounds[1][1]]
        elif self.para_len == 4:
            self.params[self.para_dic.keys()[0]] = self._randint(self.bounds[0], 4)
            self.params[self.para_dic.keys()[1]] = [self.bounds[1][0], self.bounds[1][1]]
            self.params[self.para_dic.keys()[2]] = self._randuniform(self.bounds[2], 4)
            self.params[self.para_dic.keys()[3]] = self._randuniform(self.bounds[3], 4)
        else:
            self.params[self.para_dic.keys()[0]] = [self.bounds[0][0], self.bounds[0][1]]
            self.params[self.para_dic.keys()[1]] = self._randuniform(self.bounds[1], 3)
            self.params[self.para_dic.keys()[2]] = self._randint(self.bounds[2], 3)
            self.params[self.para_dic.keys()[3]] = self._randint(self.bounds[3], 3)
            self.params[self.para_dic.keys()[4]] = self._randint(self.bounds[4], 3)


    ## Example:
    #learners_para_dic=[OrderedDict([("m",1), ("r",1),("k",1)])]
    #learners_para_bounds=[[(50,100,200, 400), (1,6), (5,21)]]
    #learners_para_categories=[["categorical", "integer", "integer"]]
    ## Paras will be keyword with default values, and bounds would be list of tuples
    def solve(self, fitness, paras=OrderedDict(), bounds=[], category=[], *r):
        self.start_time = time.time()
        self.para_len = len(paras.keys())
        self.para_dic = paras
        self.para_category = category
        self.bounds = bounds
        self.search_space()
        self.grids_making()

        population = self.pop()

        self.cur_gen = [Individual(OrderedDict(ind), fitness(ind, *r))
                        for ind in population]
        return self.early_termination_1(fitness, *r)


    def early_termination_1(self, fitness, *r):
        run_flag = True
        best_index, best_score = self._get_best_index(self.cur_gen)
        while run_flag:
            temp_pop = self.pop()
            if not temp_pop:
                run_flag = False
            else:
                trial_generation = [Individual(OrderedDict(ind), fitness(ind, *r))
                                    for ind in temp_pop]
                current_generation = self._selection(trial_generation)
                temp_best_index, temp_best_score = self._get_best_index(current_generation)
                if temp_best_score <= best_score:
                    self.termination_lifes -= 1
                else:
                    best_score = temp_best_score
                    best_index = temp_best_index
                    self.cur_gen = current_generation
                temp_duration = time.time() - self.start_time
                if temp_duration > self.eval_time or self.termination_lifes == 0:
                    run_flag = False
        tuning_time = time.time() - self.start_time
        return self.cur_gen[best_index], self.num_evals, tuning_time


    def _selection(self, trial_generation):
        generation = []
        for i in range(len(trial_generation)):
            if self.GOAL=='Max':
                if trial_generation[i].fit >= self.cur_gen[i].fit:
                    generation.append(trial_generation[i])
                else:
                    generation.append(self.cur_gen[i])
            else:
                if trial_generation[i].fit <= self.cur_gen[i].fit:
                    generation.append(trial_generation[i])
                else:
                    generation.append(self.cur_gen[i])
        return generation


    def _get_best_index(self, generation):
        if self.GOAL=='Max':
            best = 0
            max_fitness=-float("inf")
            for i, x in enumerate(generation):
                if x.fit >= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best, max_fitness
        else:
            best = 0
            max_fitness = float("inf")
            for i, x in enumerate(generation):
                if x.fit <= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best, max_fitness

    def _randint(self, a, b):
        val = a[1] + a[0]
        results = [0]*b
        for i in range(b):
            results[i] = int(math.floor((val*(i+1))/b))
        results[-1] = a[1]
        return results

    def _randchoice(self,a):
        return [a[0], a[1]]

    def _randuniform(self, a, b):
        val = a[1] + a[0]
        results = [0.0] * b
        for i in range(b):
            results[i] = float(val * (i + 1)) / float(b)
        results[-1] = a[1]
        return results

