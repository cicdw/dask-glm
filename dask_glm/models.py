from __future__ import absolute_import, division, print_function

from dask_glm.base import *
import dask.array as da
import dask.dataframe as dd
from multipledispatch import dispatch
import numpy as np
import pandas as pd
from scipy.stats import chi2

class LogisticModel(Model):

    def gradient(self,Xbeta, y):
        p = sigmoid(Xbeta)
#        p = (self.X.dot(beta)).map_blocks(sigmoid)
        return self.X.T.dot(p-y)

    def hessian(self,Xbeta, y):
        p = sigmoid(Xbeta)
#        p = (self.X.dot(beta)).map_blocks(sigmoid)
        return dot(p*(1-p)*self.X.T, self.X)

    def func(self, Xbeta, y):
        eXbeta = exp(Xbeta) # how does np.exp() interpret dask array
        return sum(log1p(eXbeta)) - dot(y, Xbeta)

    def __init__(self, X, y, **kwargs):
        super(LogisticModel, self).__init__(X, y, **kwargs)

class NormalModel(Model):

    def gradient(self,Xbeta, y):
        return self.X.T.dot(Xbeta) - self.X.T.dot(y)

    def hessian(self,Xbeta, y):
        return self.X.T.dot(self.X)

    def func(self,Xbeta, y):
        return ((y - Xbeta)**2).sum()

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
