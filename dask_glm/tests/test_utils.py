from dask_glm.utils import dot

import numpy as np
import dask.array as da


def test_dot():
    x = np.arange(6).reshape(2, 3)
    xx = da.from_array(x, chunks=x.shape)
    y = np.arange(12).reshape(3, 4)
    yy = da.from_array(y, chunks=y.shape)

    expected = x.dot(y)

    for a in [x, xx]:
        for b in [y, yy]:
            result = dot(a, b)
            if isinstance(a, da.Array) or isinstance(b, da.Array):
                assert isinstance(result, da.Array)

            result = da.compute(result)[0]
            assert np.allclose(expected, result)
