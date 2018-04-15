from __future__ import print_function, division

import sys
sys.dont_write_bytecode = True

from collections import OrderedDict, namedtuple
from random import random, randint, uniform, seed, choice, sample
import numpy as np

Individual = namedtuple('Individual', 'ind fit')

class Random(object):
    def __init__(self, NE=25, Goal="Max"):
        self.NE=NE
        self.GOAL=Goal
        #seed(1)
        #np.random.seed(1)

    def initial_pop(self):
        l=[]
        for _ in range(self.NE):
            dic=OrderedDict()
            for i in range(self.para_len):
                dic[self.para_dic.keys()[i]]=self.calls[i](self.bounds[i])
            l.append(dic)
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
        self.para_len=len(paras.keys())
        self.para_dic=paras
        self.para_category=category
        self.bounds=bounds
        self.randomisation_functions()
        initial_population=self.initial_pop()

        self.cur_gen = [Individual(OrderedDict(ind), fitness(ind, *r))
                        for ind in initial_population]
        best_index = self._get_best_index()
        return self.cur_gen[best_index], self.cur_gen

    def _get_best_index(self):
        if self.GOAL=='Max':
            best = 0
            max_fitness=-float("inf")
            for i, x in enumerate(self.cur_gen):
                if x.fit >= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best
        else:
            best = 0
            max_fitness = float("inf")
            for i, x in enumerate(self.cur_gen):
                if x.fit <= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best

    def _randint(self,a):
        return randint(*a)

    def _randchoice(self,a):
        return choice(a)

    def _randuniform(self,a):
        return uniform(*a)
