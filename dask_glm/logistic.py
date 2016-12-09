from __future__ import absolute_import, division, print_function

import dask.array as da
from multipledispatch import dispatch
import numpy as np
from scipy.stats import chi2

def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1/(1+da.exp(-x))

@dispatch(np.ndarray,np.ndarray)
def dot(A,B):
    return np.dot(A,B)

@dispatch(da.Array,da.Array)
def dot(A,B):
    return da.dot(A,B)

class Optimizer(object):

    def initialize(self, size, value=None, method=None):
        '''Method for setting the initialization.'''

        if value:
            initial = value
        else:
            initial = np.zeros(size)

        return initial

    def hessian(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError

    def func(self):
        raise NotImplementedError

    def bfgs(self, verbose=True, max_steps=100):
        recalcRate = 10
        stepSize = 1.0
        stepGrowth = 1.25
        beta = self.init
        M = beta.shape[0]
        Hk = np.eye(M)

        if verbose:
            print('##       -f        |df/f|    step')
            print('----------------------------------------------')

        for k in range(max_steps):

            if k % recalcRate==0:
                Xbeta = self.X.dot(beta)
                func = self.func(Xbeta)

            gradient = self.gradient(Xbeta)

            if k:
                yk += gradient
                rhok = 1/yk.dot(sk)
                adj = np.eye(M) - rhok*sk.dot(yk.T)
                Hk = adj.dot(Hk.dot(adj.T)) + rhok*sk.dot(sk.T)

            step = Hk.dot(gradient)
            steplen = step.dot(gradient)
            Xstep = self.X.dot(step)

            Xbeta, func, steplen, step, Xstep = da.compute(
                    Xbeta, func, steplen, step, Xstep)

            # Compute the step size
            if k==0:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, step, Xstep,
                    stepSize, steplen, **{'backtrackMult' : 0.1,
                        'armijoMult' : 1e-4})
            else:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, step, Xstep,
                    stepSize, steplen, **{'armijoMult' : 1e-4})

            yk = -gradient
            sk = -stepSize*step
            stepSize = 1.0
            df = func-fnew
            func = fnew

            if stepSize == 0:
                if verbose:
                    print('No more progress')

            df /= max(func, fnew)
            if verbose:
                print('%2d  %.6e %9.2e  %.1e' % (k + 1, func, df, stepSize))
            if df < 1e-14:
                print('Converged')
                break

        return beta

    def _check_convergence(self, old, new, tol=1e-4, method=None):
        coef_change = np.absolute(old - new)
        return not np.any(coef_change>tol)

    def fit(self, X, y, method=None, **kwargs):
        raise NotImplementedError

    def _newton_step(self,curr,Xcurr):

        hessian = self.hessian(Xcurr)
        grad = self.gradient(Xcurr)

        # should this be dask or numpy?
        step, *_ = da.linalg.lstsq(hessian, grad)
        beta = curr - step
        
        return beta.compute()

    def newton(self):
    
        beta = self.init
        Xbeta = self.X.dot(beta)

        iter_count = 0
        converged = False

        while not converged:
            beta_old = beta
            beta = self._newton_step(beta,Xbeta)
            Xbeta = self.X.dot(beta)
            iter_count += 1
            
            converged = (self._check_convergence(beta_old, beta) & (iter_count<self.max_iter))

        return beta

    def gradient_descent(self, max_steps=100, verbose=True):
        recalcRate = 10
        stepSize = 1.0
        stepGrowth = 1.25
        beta = self.init

        if verbose:
            print('##       -f        |df/f|    step')
            print('----------------------------------------------')

        for k in range(max_steps):

            if k % recalcRate==0:
                Xbeta = self.X.dot(beta)
                func = self.func(Xbeta)

            gradient = self.gradient(Xbeta)
            steplen = (gradient**2).sum()
            Xgradient = self.X.dot(gradient)

            Xbeta, func, gradient, steplen, Xgradient = da.compute(
                    Xbeta, func, gradient, steplen, Xgradient)

            # Compute the step size
            if k==0:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, gradient, Xgradient,
                    stepSize, steplen, **{'backtrackMult' : 0.1})
            else:
                stepSize, beta, Xbeta, fnew = self._backtrack(func,
                    beta, Xbeta, gradient, Xgradient,
                    stepSize, steplen)

            stepSize *= stepGrowth
            df = func-fnew
            func = fnew

            if stepSize == 0:
                if verbose:
                    print('No more progress')

            df /= max(func, fnew)
            if verbose:
                print('%2d  %.6e %9.2e  %.1e' % (k + 1, func, df, stepSize))
            if df < 1e-14:
                print('Converged')
                break

        return beta

    ## this is currently specific to linear models
    def _backtrack(self, curr_val, curr, Xcurr, 
        step, Xstep, stepSize, steplen, **kwargs):

        ## theres got to be a better way...
        params = {'backtrackMult' : 0.5,
            'armijoMult' : 0.1}

        params.update(kwargs)
        backtrackMult = params['backtrackMult']
        armijoMult = params['armijoMult']
        Xbeta = Xcurr

        for ii in range(100):
            beta = curr - stepSize*step

            if ii and np.array_equal(curr, beta):
                stepSize = 0
                break
            Xbeta = Xcurr - stepSize*Xstep

            func = self.func(Xbeta)
            df = curr_val - func
            if df >= armijoMult * stepSize * steplen:
                break
            stepSize *= backtrackMult

        return stepSize, beta, Xbeta, func

    def __init__(self, max_iter=50, init_type='zeros'):
        self.max_iter = 50

class Model(Optimizer):
    '''Class for holding all output statistics.'''

    def fit(self,method='newton',**kwargs):
        self.coefs = self.newton(self.X, self.y)

    def pvalues(self, names={}):
        H = self.hessian(self.X.dot(self.coefs))
        covar = np.linalg.inv(H.compute())
        variance = np.diag(covar)
        self.chi = self.coefs**2 / variance

    def summary(self):
        raise NotImplementedError

    def __init__(self):
        return None

class LogisticModel(Optimizer):

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
        super(LogisticModel, self).__init__(**kwargs)
        self.X, self.y = X, y
        self.init = self.initialize(X.shape[1])

class NormalModel(Optimizer):

    def gradient(self,Xbeta):
        return self.X.T.dot(Xbeta) - self.X.T.dot(self.y)

    def hessian(self,Xbeta):
        return self.X.T.dot(self.X)

    def func(self,Xbeta):
        return ((self.y - Xbeta)**2).sum()

    def __init__(self, X, y, **kwargs):
        super(NormalModel, self).__init__(**kwargs)
        self.X, self.y = X, y
        self.init = self.initialize(X.shape[1])
