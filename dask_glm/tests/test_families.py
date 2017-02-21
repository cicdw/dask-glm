import pytest

from dask import persist
import numpy as np
import dask.array as da

from dask_glm.algorithms import (newton, bfgs, proximal_grad,
                                 gradient_descent, admm)
from dask_glm.families import Normal, Logistic


@pytest.mark.parametrize('family', [Normal, Logistic])
@pytest.mark.parametrize('algorithm', [newton,
                                       gradient_descent,
                                       admm,
                                       # proximal_grad,
])
def test_basic(family, algorithm):
    X = np.array([[1, 1], [2, 1], [3, 1]])
    y = np.array([1, 2, 3])

    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))

    beta = algorithm(dX, dy, family=family)
    assert isinstance(beta, np.ndarray)
