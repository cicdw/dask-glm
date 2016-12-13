from __future__ import absolute_import, division, print_function

from dask_glm.base import *
import dask.array as da
import dask.dataframe as dd
from multipledispatch import dispatch
import numpy as np
import pandas as pd
from scipy.stats import chi2

class LogisticModel(Model):

    def gradient(self,Xbeta):
        p = sigmoid(Xbeta)
#        p = (self.X.dot(beta)).map_blocks(sigmoid)
        return self.X.T.dot(p-self.y)

    def hessian(self,Xbeta):
        p = sigmoid(Xbeta)
#        p = (self.X.dot(beta)).map_blocks(sigmoid)
        return dot(p*(1-p)*self.X.T, self.X)

    def func(self,Xbeta):
        eXbeta = np.exp(Xbeta)
        return np.sum(np.log1p(eXbeta)) - np.dot(self.y, Xbeta)

    def negloglike(self,beta):
        Xbeta = self.X.dot(beta)
        eXbeta = da.exp(Xbeta)
        return da.log1p(eXbeta).sum() - self.y.dot(Xbeta)

    def __init__(self, X, y, **kwargs):
        super(LogisticModel, self).__init__(X, y, **kwargs)

class NormalModel(Model):

    def gradient(self,Xbeta):
        return self.X.T.dot(Xbeta) - self.X.T.dot(self.y)

    def hessian(self,Xbeta):
        return self.X.T.dot(self.X)

    def func(self,Xbeta):
        return ((self.y - Xbeta)**2).sum()

    def __init__(self, X, y, **kwargs):
        super(NormalModel, self).__init__(X, y, **kwargs)

class PoissonModel(Model):

    def gradient(self,Xbeta):
        raise NotImplementedError

    def hessian(self,Xbeta):
        raise NotImplementedError

    def func(self,Xbeta):
        raise NotImplementedError

    def __init__(self, X, y, **kwargs):
        super(NormalModel, self).__init__(**kwargs)
        self.X, self.y = X, y
