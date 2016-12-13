from dask_glm.models import Optimizer
import numpy as np
import unittest

class TestOptimizer(unittest.TestCase):
    '''Testing Optimizer Class.'''

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
