import dask.array as da
import dask.dataframe as dd
from dask_glm.models import NormalModel
import numpy as np
import unittest


def generate_2pt_line():
    '''Generates trivial data.'''
    X = da.from_array(np.array([[0], [1]]), chunks=2)
    y = da.from_array(np.array([[0], [1]]), chunks=2)
    return y, X


class TestNormal(unittest.TestCase):
    '''Testing Normal Model Class.'''

    def test_array_input(self):
        '''Testing array inputs.'''
        y, X = generate_2pt_line()
        model = NormalModel(X=X, y=y)
        assert model.X.shape == (2, 1)

    def test_dataframe_input(self):
        '''Testing dataframe inputs.'''
        y, X = generate_2pt_line()
        A = dd.from_dask_array(X)
        A.columns = ['var1']
        A = A.assign(y=dd.from_dask_array(y[:, 0]))
        model = NormalModel(X=A[['var1']], y=A[['y']])
        assert model.X.shape[1] == 1

    def test_series_input(self):
        '''Testing Series inputs.'''
        y, X = generate_2pt_line()
        A = dd.from_dask_array(X)
        A.columns = ['var1']
        A = A.assign(y=dd.from_dask_array(y[:, 0]))
        model = NormalModel(X=A['var1'], y=A['y'])
        assert model.names[0] == 'var1'

    def test_gradient_fit(self):
        '''Testing gradient_descent fit with Series input.'''
        y, X = generate_2pt_line()
        A = dd.from_dask_array(X)
        A.columns = ['var1']
        A = A.assign(y=dd.from_dask_array(y[:, 0]))
        model = NormalModel(X=A[['var1']], y=A['y'])
        model = model.fit(method='gradient_descent')
        assert np.isclose(model.coefs[0], 1.0)

    def test_naive_fit(self):
        '''Testing simple linear fit with Series input.'''
        y, X = generate_2pt_line()
        A = dd.from_dask_array(X)
        A.columns = ['var1']
        A = A.assign(y=dd.from_dask_array(y[:, 0]))
        model = NormalModel(X=A['var1'], y=A['y'])
        model = model.fit()
        assert np.isclose(model.coefs[0], 1.0)

    def test_pval_simple_fit(self):
        '''Testing p-values for simple linear fit.'''
        y, X = generate_2pt_line()
        A = dd.from_dask_array(X)
        A.columns = ['var1']
        A = A.assign(y=dd.from_dask_array(y[:, 0]))
        model = NormalModel(X=A[['var1']], y=A[['y']])
        model = model.fit()
        assert np.isclose(model.se[0], 1)
        assert np.isclose(model.pvals[0], 0.31731051)

    def test_simple_fit(self):
        '''Testing simple linear fit.'''
        y, X = generate_2pt_line()
        A = dd.from_dask_array(X)
        A.columns = ['var1']
        A = A.assign(y=dd.from_dask_array(y[:, 0]))
        model = NormalModel(X=A[['var1']], y=A[['y']])
        model = model.fit()
        assert np.isclose(model.coefs[0], 1.0)
