import numpy as np
from math import sin, exp, sqrt, e
import pytest
from scipy.linalg import hilbert

from Solvers.DiffEqSolver import DiffEquationSolver
from Solvers.LU_decompositor import LUDecomposer


def test_simple():
    p = lambda x: x - 1
    q = lambda x: -x
    r = lambda x: 1
    f = lambda x: (x - 1) ** 2
    a, b = 0, 0.5
    alpha, beta = 0, 0
    true_func = lambda x: -1 + exp(x) + (5 * x) / 2 - 2 * sqrt(e) * x - x ** 2
    true_func = np.vectorize(true_func)

    diffEqSolver = DiffEquationSolver(p, q, r, f, a, b, alpha, beta, )
    answ, R_l_2, R_C, i = diffEqSolver.solve(0.000001, 2)
    true_table = true_func(diffEqSolver.cur_grid).reshape(len(diffEqSolver.cur_grid), 1)
    np.testing.assert_allclose(answ, true_table, atol=1e-2)



