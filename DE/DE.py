from __future__ import print_function, division

import sys
sys.dont_write_bytecode = True

from collections import OrderedDict, namedtuple
from random import random, randint, uniform, seed, choice, sample
import numpy as np
import math
import time
import pdb

__all__ = ['DE']
Individual = namedtuple('Individual', 'ind fit')

class DE(object):
    def __init__(self, F=0.3, CR=0.7, NP=10, GEN=0, Goal="Max", termination="Early", num_lifes=5, eval_time=600):
        self.F=F
        self.CR=CR
        self.NP=NP
        self.GEN=GEN
        self.GOAL=Goal
        self.termination=termination
        self.eval_time=eval_time
        self.termination_lifes=num_lifes
        self.start_time=0
        self.num_evals=0

    def initial_pop(self):
        l=[]
        for _ in range(self.NP):
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
        self.start_time = time.time()
        self.para_len=len(paras.keys())
        self.para_dic=paras
        self.para_category=category
        self.bounds=bounds
        self.randomisation_functions()
        self.cur_gen = []
        initial_population=self.initial_pop()
        for individual in initial_population:
            individual_score = fitness(individual, *r)
            self.cur_gen.append(Individual(OrderedDict(individual), individual_score))
        if self.termination=='Early':
            return self.early_termination_1(fitness,*r)
        else:
            return self.late_termination(fitness,*r)


    def early_termination_1(self,fitness,*r):
        run_flag = True
        while run_flag:
            temp = []
            for i in range(len(self.cur_gen)):
                v = self._extrapolate(self.cur_gen[i])
                trial_ind = Individual(OrderedDict(v), fitness(v, *r))
                val_temp = self._selection_per_eval(trial_ind, self.cur_gen[i])
                if math.isnan(val_temp.fit):
                    temp.append(Individual(val_temp.ind, 0.0))
                else:
                    temp.append(Individual(val_temp.ind, val_temp.fit))

                if self.num_evals > 0 and i > 0:
                    if temp[i].fit <= temp[i-1].fit:
                        self.termination_lifes -= 1
                self.num_evals += 1
                temp_duration = time.time() - self.start_time
                if temp_duration > self.eval_time or self.termination_lifes == 0:
                    run_flag = False
                    break
            self.cur_gen = temp
        best_index = self._get_best_index()
        tuning_time = time.time() - self.start_time
        return self.cur_gen[best_index], self.num_evals, tuning_time


    def early_termination(self,fitness,*r):
        for x in self.GEN:
            trial_generation = []
            for ind in self.cur_gen:
                v = self._extrapolate(ind)
                trial_generation.append(Individual(OrderedDict(v), fitness(v, *r)))

            current_generation = self._selection(trial_generation)
            self.cur_gen=current_generation
        best_index = self._get_best_index()
        return self.cur_gen[best_index], self.cur_gen

    def late_termination(self,fitness,*r):
        lives=1
        while lives!=0:
            trial_generation = []
            for ind in self.cur_gen:
                v = self._extrapolate(ind)
                #print(v)
                trial_generation.append(Individual(OrderedDict(v), fitness(v,*r)))
            current_generation = self._selection(trial_generation)
            if sorted(self.cur_gen)==sorted(current_generation):
                lives=lives-1
            else:
                self.cur_gen=current_generation

        best_index = self._get_best_index()
        return self.cur_gen[best_index], self.cur_gen

    def _extrapolate(self,ind):
        if (random() < self.CR):
            l = self._select3others()
            mutated=[]
            for x,i in enumerate(self.para_category):
                if i=='continuous':
                    mutated.append(l[0][l[0].keys()[x]]+self.F*(l[2][l[2].keys()[x]]-l[2][l[2].keys()[x]]))
                else:
                    mutated.append(self.calls[x](self.bounds[x]))
            check_mutated = []
            for i in range(self.para_len):
                if self.para_category[i]=='continuous':
                    check_mutated.append(max(self.bounds[i][0], min(mutated[i], self.bounds[i][1])))
                else:
                    check_mutated.append(mutated[i])
            dic=OrderedDict()
            for i in range(self.para_len):
                dic[self.para_dic.keys()[i]]=check_mutated[i]
            return dic
        else:
            dic = OrderedDict()
            for i in range(self.para_len):
                key=self.para_dic.keys()[i]
                dic[self.para_dic.keys()[i]] = ind.ind[key]
            return dic

    def _select3others(self):
        l=[]
        val=sample(self.cur_gen,3)
        for a in val:
            l.append(a.ind)
        return l

    def _selection_per_eval(self, a, b):
        if self.GOAL=='Max':
            if a.fit >= b.fit:
                return a
            else:
                return b
        else:
            if a.fit <= b.fit:
                return a
            else:
                return b

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

    def _get_best_index(self):
        if self.GOAL=='Max':
            best = 0
            max_fitness=-float("inf")
            for i, x in enumerate(self.cur_gen):
                val_xfit = x.fit
                if math.isnan(val_xfit):
                    val_xfit = 0
                if val_xfit >= max_fitness:
                    best = i
                    max_fitness = val_xfit
            return best
        else:
            best = 0
            max_fitness = float("inf")
            for i, x in enumerate(self.cur_gen):
                if math.isnan(x.fit):
                    x.fit = 0
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
