from __future__ import absolute_import, division, print_function

from dask_glm.base import *
import dask.array as da
import dask.dataframe as dd
from multipledispatch import dispatch
import numpy as np
import pandas as pd
from scipy.stats import chi2

def l2_prox(beta, t):
    return beta/(1+t)

def l1_prox(beta, t):
    return (np.abs(beta)>t)*np.sign(beta)*(np.abs(beta) - t)

prox_dict = {'l2' : l2_prox,
    'l1' : l1_prox}

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
        eXbeta = exp(Xbeta) + 1
        return sum(log1p(eXbeta)) - dot(y, Xbeta)

    def __init__(self, X, y, reg=None, **kwargs):
        super(LogisticModel, self).__init__(X, y, **kwargs)

        if reg:
            self.prox = prox_dict[reg]

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

class L2(Prior):

    def gradient(self, beta):
        return self.lam * beta

    def hessian(self, beta):
        ## assumes few variables so this is "efficient"
        return self.lam * np.eye(beta.shape[0])

    def func(self, beta):
        return (self.lam / 2) * (beta**2).sum()

    def prox(self, beta):
        raise NotImplementedError

    def __init__(self, lam=0.1):
        
        self.lam = lam
        return self

