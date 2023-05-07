import numpy as np
import pytest
from scipy.linalg import hilbert
from Solvers.LU_decompositor import LUDecomposer


@pytest.mark.parametrize("size", [i for i in range(2, 10)])
def test_decomp_on_Hilbert(size):
    H = hilbert(size)
    lu = LUDecomposer(H)
    L, U = lu.decompose()
    result = L @ U
    np.testing.assert_allclose(H, result)


@pytest.mark.parametrize("size", [i for i in range(2, 7)])
def test_solve_on_Hilbert(size):
    H = hilbert(size)
    lu = LUDecomposer(H)
    lu.decompose()
    x = np.ones((size, 1)).flatten()
    b = (H @ x).flatten()
    lu_x = lu.solve(b)
    np.testing.assert_allclose(x, lu_x)


@pytest.mark.parametrize("size", [i for i in range(7, 12)])
def test_solve_on_big_Hilbert(size):
    H = hilbert(size)
    x = np.ones((size, 1)).flatten()
    b = (H @ x).flatten()
    lu = LUDecomposer(H)
    lu.decompose()
    lu_x = lu.solve(b)
    np.testing.assert_allclose(x, lu_x)