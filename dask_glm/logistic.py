from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch
import numpy as np
import dask.array as da

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
        if value:
            initial = value
        else:
            initial = np.zeros(size)

        return initial

    def hessian(self,x):
        raise NotImplementedError

    def gradient(self,x):
        raise NotImplementedError

    def func(self,x):
        raise NotImplementedError

    def _bfgs_step(self,curr):
        raise NotImplementedError

    def bfgs(self, curr):
        raise NotImplementedError

    def _check_convergence(self, old, new, tol=1e-4, method=None):
        coef_change = np.absolute(old - new)
        return not np.any(coef_change>tol)

    def fit(self, X, y, method=None, **kwargs):
        raise NotImplementedError

    def __init__(self, max_iter=50):
        self.max_iter = 50

    def _newton_step(self,curr):

        hessian = self.hessian(curr)
        grad = self.gradient(curr)

        step, *_ = da.linalg.lstsq(hessian, grad)
        beta = curr + step
        
        return beta.compute()

    def newton(self):
    
        beta = self.init

        iter_count = 0
        converged = False

        while not converged:
            beta_old = beta
            beta = self._newton_step(beta)
            iter_count += 1
            
            converged = (self._check_convergence(beta_old, beta) & (iter_count<self.max_iter))

        return beta

    def gradient_descent(self, X, y, max_steps=100, verbose=True):
        recalcRate = 10
        stepSize = 1.0
        stepGrowth = 1.25
        beta = self.init

        if verbose:
            print('##       -f        |df/f|    step')
            print('----------------------------------------------')

        for k in range(max_steps):

            Xbeta = X.dot(beta)
            func = self.func(Xbeta)
            gradient = -self.gradient(beta)
            steplen = (gradient**2).sum()**0.5
            Xgradient = X.dot(gradient)

            Xbeta, func, gradient, steplen, Xgradient = da.compute(
                    Xbeta, func, gradient, steplen, Xgradient)

            # Compute the step size
            if k:
                stepSize, beta, fnew = self._backtrack(func,
                    beta, Xbeta, gradient, Xgradient,
                    stepSize, steplen, **{'backtrackMult' : 0.5})
            else:
                stepSize, beta, fnew = self._backtrack(func,
                    beta, Xbeta, gradient, Xgradient,
                    stepSize, steplen, **{'backtrackMult' : 0.1})

            stepSize *= stepGrowth
            df = func-fnew

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

        for ii in range(100):
            beta = curr - stepSize*step

            if ii and np.array_equal(curr, beta):
                stepSize = 0
                break

            Xbeta = Xcurr - stepSize*Xstep

            func = self.func(Xbeta)
            df = curr_val - func
            if df >= armijoMult * stepSize * steplen ** 2:
                break
            stepSize *= backtrackMult

        return stepSize, beta, func

class LogisticModel(Optimizer):

    def gradient(self, beta):
        p = (self.X.dot(beta)).map_blocks(sigmoid)
        return self.X.T.dot(self.y-p)

    def hessian(self, beta):
        p = (self.X.dot(beta)).map_blocks(sigmoid)
        return dot(p*(1-p)*self.X.T, self.X)

    def func(self,Xbeta):
        eXbeta = np.exp(Xbeta)
        return np.sum(np.log1p(eXbeta)) - np.dot(self.y, Xbeta)

    def loglike(self,beta):
        Xbeta = self.X.dot(beta)
        eXbeta = da.exp(Xbeta)
        return da.log1p(eXbeta).sum() - self.y.dot(Xbeta)

    def __init__(self, X, y, **kwargs):
        super(LogisticModel, self).__init__(**kwargs)
        self.X, self.y = X, y
        self.init = self.initialize(X.shape[1])
