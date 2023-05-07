import numpy as np
from numpy import testing
import pytest
from scipy.linalg import hilbert
from Solvers.MaxEigen import EigenMax
from Utils.Generator import *
from itertools import product

@pytest.mark.parametrize("size, eps, dec", [(i, 10 ** (-eps), eps) for i, eps in product(range(2, 8), range(2, 6))])
def test_eigen_on_power(size, eps, dec):
    A = generate_random(size)
    eigMax = EigenMax(A)
    eigs = np.linalg.eigvals(A)
    eig = max(abs(eigs))

    alg_eig, actual_eigvec, _ = eigMax.power_method(eps)
    testing.assert_approx_equal(eig, alg_eig, significant=dec)


@pytest.mark.parametrize("size, eps, dec", [(i, 10 ** (-eps), eps) for i, eps in product(range(2, 8), range(2, 6))])
def test_eigen_on_power(size, eps, dec):
    A = generate_random(size)
    eigMax = EigenMax(A)
    eigs = np.linalg.eigvals(A)
    eig = max(abs(eigs))

    alg_eig, actual_eigvec, _ = eigMax.dot_prod_method(eps)
    testing.assert_approx_equal(eig, alg_eig, significant=dec)