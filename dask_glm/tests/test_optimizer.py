from __future__ import division, print_function
from dask_glm.models import Optimizer
import numpy as np
import unittest

class QuadraticTest(Optimizer):
    '''Testing Optimizer through simple quadratic function.'''

    def func(self, x):
        return 0.5*(x-3)**2

    def hessian(self, x):
        return np.ones(x.shape[0])

    def gradient(self, x):
        return x-3
    
    def __init__(self, size):
        self.max_iter = 50
        self = self.initialize(size)

class TestOptimizer(unittest.TestCase):
    '''Testing Optimizer Class.'''

    def test_quadratic_backtrack(self):
        '''Testing backtracking line search with univariate quadratic.'''
        quad = QuadraticTest(1)
        steplen = (quad.gradient(quad.init)**2).sum()
        alpha = 0.5
        
        ## first test all armijo multipliers less than 3/4
        for armijo in [0,0.25,0.5,0.74]:
            stepSize, beta, Xbeta, func = quad._backtrack(
                9/2, quad.init, quad.init, quad.gradient(quad.init),
                quad.gradient(quad.init), stepSize = alpha, 
                steplen = steplen, **{'armijoMult' : armijo})
            assert stepSize==alpha
            assert beta==3/2
            assert Xbeta==3/2
            assert func==9/8

        ## next test all armijo multipliers larger than 3/4
        armijo = 7/8
        for power in [0,1,2]:
        
            gamma = 0.5**power

            stepSize, beta, Xbeta, func = quad._backtrack(
                9/2, quad.init, quad.init, quad.gradient(quad.init),
                quad.gradient(quad.init), stepSize = alpha, 
                steplen = steplen, **{'armijoMult' : armijo,
                    'backtrackMult' : gamma})
            assert stepSize==gamma*alpha
            assert beta==3*gamma*alpha
            assert Xbeta==beta
            assert func==0.5*(beta-3)**2

    def test_default_initialization(self):
        '''Testing default initialization routine.'''

        opt = Optimizer()
        for i in [1,50]:
            opt = opt.initialize(i)
            assert np.all(np.isclose(opt.init, np.zeros(i)))

    def test_random_initialization(self):
        '''Testing random initialization routine.'''

        opt = Optimizer()
        for i in [1,50]:
            opt = opt.initialize(i,method='random')
            assert opt.init.shape == (i,)

if __name__=='__main__':
    unittest.main()
