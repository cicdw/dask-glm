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

class Logistic(object):

    def initialize(self, size, value=None, method=None):
        if value:
            self.init = value
        else:
            self.init = np.zeros(size)

        return self.init

    def _check_convergence(self, old, new, tol=1e-4, method=None):
        
        coef_change = np.absolute(old - new)
        return not np.any(coef_change>tol)
        
    def fit(self, method=None, **kwargs):
        raise NotImplementedError

    def __init__(self, max_iter=50):
        self.max_iter = 50

    def _newton_step(self,curr,X,y):
        p = (X.dot(curr)).map_blocks(sigmoid)
        hessian = dot(p*(1-p)*X.T, X)
        grad = X.T.dot(y-p)
        step, *_ = da.linalg.lstsq(hessian, grad)
        beta = curr + step
        
        return beta

    def newton(self,X,y):
        beta = self.initialize(X.shape[1])
        iter_count = 0
        converged = False

        while not converged:
            beta_old = beta
            beta = self._newton_step(beta,X,y)
            iter_count += 1
            
            converged = (self._check_convergence(old, new) & (iter_count<self.max_iter))

        return beta

    def gradient(X, y, max_steps=100):
        N, M = X.shape
        firstBacktrackMult = 0.1
        nextBacktrackMult = 0.5
        armijoMult = 0.1
        stepGrowth = 1.25
        stepSize = 1.0
        recalcRate = 10
        backtrackMult = firstBacktrackMult
        beta = np.zeros(M)

        print('##       -f        |df/f|    |dx/x|    step')
        print('----------------------------------------------')
        for k in range(max_steps):
            # Compute the gradient
            if k % recalcRate == 0:
                Xbeta = X.dot(beta)
                eXbeta = da.exp(Xbeta)
                func = da.log1p(eXbeta).sum() - y.dot(Xbeta)
            e1 = eXbeta + 1.0
            gradient = X.T.dot(eXbeta / e1 - y)
            steplen = (gradient**2).sum()**0.5
            Xgradient = X.dot(gradient)

            Xbeta, eXbeta, func, gradient, steplen, Xgradient = da.compute(
                    Xbeta, eXbeta, func, gradient, steplen, Xgradient)

            obeta = beta
            oXbeta = Xbeta

            # Compute the step size
            lf = func
            for ii in range(100):
                beta = obeta - stepSize * gradient
                if ii and np.array_equal(beta, obeta):
                    stepSize = 0
                    break
                Xbeta = oXbeta - stepSize * Xgradient
                # This prevents overflow
                if np.all(Xbeta < 700):
                    eXbeta = np.exp(Xbeta)
                    func = np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)
                    df = lf - func
                    if df >= armijoMult * stepSize * steplen ** 2:
                        break
                stepSize *= backtrackMult
            if stepSize == 0:
                print('No more progress')
                break
            df /= max(func, lf)
            db = stepSize * steplen / (np.linalg.norm(beta) + stepSize * steplen)
            print('%2d  %.6e %9.2e  %.2e  %.1e' % (k + 1, func, df, db, stepSize))
            if df < 1e-14:
                print('Converged')
                break
            stepSize *= stepGrowth
            backtrackMult = nextBacktrackMult

        return beta
