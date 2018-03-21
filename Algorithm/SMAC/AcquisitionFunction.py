from __future__ import division
import numpy as np
from scipy.stats import norm

class AbstractAcquisitionFunction(object):
    def __init__(self, model):
        self.model = model

    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, configurations):
        acq = self._compute(configurations)
        return acq

    def _compute(self, X):
        raise NotImplementedError



class LogEI(AbstractAcquisitionFunction):

    def __init__(self, model, par=0.0, **kwargs):
        r"""Computes for a given x the logarithm expected improvement as
        acquisition value.
        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LogEI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X, **kwargs):
        """Computes the EI value and its derivatives.
        Parameters
        ----------
        X:  The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D), with
         N as the number of points to evaluate at and D is the number of dimensions of one X.
        Returns
        -------
            Expected Improvement of X
        """

        means, variances = self.model.predict(X)
        std = np.sqrt(variances)

        f_min = self.eta - self.par
        v = (np.log(f_min) - means) / std
        log_ei = (f_min * norm.cdf(v)) - (np.exp(0.5 * variances + means) * norm.cdf(v - std))

        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            print("Predicted std is 0.0 for at least one sample.")
            log_ei[std == 0.0] = 0.0

        if (log_ei < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))