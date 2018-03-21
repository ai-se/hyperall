class smbo:
    def __init__(self, configuration_space, initial_design, intensifier, model, runhistory, acq_optimizer, acquition_func):
        self.configuration_space = configuration_space
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.acq_optimizer = acq_optimizer  # InterleavedLocalAndRandomSearch
        self.acq_function = acquition_func  # LogEI
        self.model = model
        self.incumbent = None
        self.intensifier = intensifier

    def choose_next(self, independents, dependents):
        self.model.train(independents, dependents)
        if self.incumbent is None:
            print "Something is wrong"
            exit()

        self.acq_function.update(model=self.model, eta=self.incumbent)
        challengers = self.acq_optimizer.maximize(self.runhistory, int(self.configuration_space.get_configuration_size()*0.1))
        return challengers


    def start(self):
        if self.incumbent is None:
            # TODO: incumbent must be of Solution type
            incumbent_indep = self.initial_design.run()
            assert(len(incumbent_indep) == 1), "Something is wrong"
            incumbent_indep = incumbent_indep[-1]
            self.incumbent = self.configuration_space.get_performance(incumbent_indep)
            print incumbent_indep, self.incumbent
            self.runhistory.add(incumbent_indep, self.incumbent)

    def run(self):
        self.start()

        while True:
            X, Y = self.runhistory.get_candidates()
            assert(len(X) == len(Y)), "Something is wrong"
            challengers = self.choose_next(X, Y)

            # Choose the challenger which has the highest value (acquisition function)
            chosen = challengers[0]

            self.runhistory.add(chosen, self.configuration_space.get_performance(chosen))

            self.incumbent = self.runhistory.get_incumbent()
            # self.incumbent, inc_perf = self.intensifier.intensify(challengers=challengers, incumbent=self.incumbent, run_history=self.runhistory)

            if self.stats.is_budget_exhausted(): break

        return self.incumbent

