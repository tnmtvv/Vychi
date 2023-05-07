import numpy as np
import pytest
from scipy.linalg import hilbert
from Solvers.Jacobi import JacobiEigen
from Utils.Generator import *

@pytest.mark.parametrize("size", [i for i in range(2, 20)])
def test_Jacobi(size):
    A = generate_symmetric(size)
    eps = 0.000000001
    expected = np.linalg.eigvalsh(A)
    jacEigen = JacobiEigen(A, eps=eps)
    alg, rotations = jacEigen.eigenvals, jacEigen.iterations
    expected.sort()
    alg.sort()
    np.testing.assert_allclose(expected, alg, rtol=eps)





