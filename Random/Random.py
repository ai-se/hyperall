from __future__ import print_function, division

import sys
import time
sys.dont_write_bytecode = True

from collections import OrderedDict, namedtuple
from random import random, randint, uniform, seed, choice, sample
import numpy as np

Individual = namedtuple('Individual', 'ind fit')

class Random(object):
    def __init__(self, NP=10, Goal="Max", num_lifes=5, eval_time=600):
        self.NP=NP
        self.GOAL=Goal
        self.eval_time = eval_time
        self.termination_lifes = num_lifes
        self.start_time = 0
        self.num_evals = 0
        #seed(1)
        #np.random.seed(1)

    def initial_pop(self):
        l=[]
        for _ in range(self.NP):
            dic=OrderedDict()
            for i in range(self.para_len):
                dic[self.para_dic.keys()[i]] = self.calls[i](self.bounds[i])
            l.append(dic)
        self.num_evals += 1
        return l


    ## Need a tuple for integer and continuous variable but need the whole list for category
    def randomisation_functions(self):
        l=[]
        for i in self.para_category:
            if i=='integer':
                l.append(self._randint)
            elif i=='continuous':
                l.append(self._randuniform)
            elif i=='categorical':
                l.append(self._randchoice)
        self.calls=l

    ## Example:
    #learners_para_dic=[OrderedDict([("m",1), ("r",1),("k",1)])]
    #learners_para_bounds=[[(50,100,200, 400), (1,6), (5,21)]]
    #learners_para_categories=[["categorical", "integer", "integer"]]
    ## Paras will be keyword with default values, and bounds would be list of tuples
    def solve(self, fitness, paras=OrderedDict(), bounds=[], category=[], *r):
        self.start_time = time.time()
        self.para_len=len(paras.keys())
        self.para_dic=paras
        self.para_category=category
        self.bounds=bounds
        self.randomisation_functions()
        initial_population=self.initial_pop()

        self.cur_gen = [Individual(OrderedDict(ind), fitness(ind, *r))
                        for ind in initial_population]
        return self.early_termination_1(fitness,*r)


    def early_termination_1(self,fitness,*r):
        run_flag = True
        best_index, best_score = self._get_best_index(self.cur_gen)
        while run_flag:
            temp_pop = self.initial_pop()
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

        for a, b in zip(self.cur_gen, trial_generation):
            if self.GOAL=='Max':
                if a.fit >= b.fit:
                    generation.append(a)
                else:
                    generation.append(b)
            else:
                if a.fit <= b.fit:
                    generation.append(a)
                else:
                    generation.append(b)
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

    def _randint(self,a):
        return randint(*a)

    def _randchoice(self,a):
        return choice(a)

    def _randuniform(self,a):
        return uniform(*a)
