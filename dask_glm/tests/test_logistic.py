import pytest

from dask import persist
import numpy as np
import dask.array as da

from dask_glm.logistic import (newton, bfgs, proximal_grad,
                               gradient_descent, admm)
from dask_glm.utils import sigmoid, make_y


@pytest.mark.parametrize('func,kwargs', [
    (newton, {'tol': 1e-8}),
    (gradient_descent, {'tol': 1e-14}),
    (admm, {'lamduh': 0, 'abstol': 1e-14, 'reltol': 1e-14}),
])
@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('nchunks', [1, 5])
@pytest.mark.parametrize('p', [1, 5, 10])
def test_unregularized_accuracy(func, kwargs, N, p, nchunks):
    X = da.random.random((N, p), chunks=(N // nchunks, p))
    beta = 2*np.random.random(p)
    y = sigmoid(X.dot(beta))

    X, y = persist(X, y)

    result = func(X, y, **kwargs)

    assert np.allclose(result, beta, atol=1e-2)
