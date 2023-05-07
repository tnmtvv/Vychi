import numpy as np
import pytest
from scipy.linalg import hilbert
from Solvers.IterativeSolver import IterativeSolver
from Utils.Generator import *


@pytest.mark.parametrize("size", [i for i in range(2, 12)])
def test_solve_on_Hilbert(size):
    H = hilbert(size)
    x = np.ones((size, 1)).flatten()
    b = (H @ x).flatten()
    iterSolve = IterativeSolver(H, b)
    alg_x = list(map(lambda x: round(x), list(iterSolve.solve(if_pos_definitive=True, eps=0.000001, get_iterations=True))))
    np.testing.assert_allclose(x, alg_x)


@pytest.mark.parametrize("size", [i for i in range(3, 12)])
def test_solve_on_diag_prevail(size):
    H = generate_diag_prevail(size)
    x = np.ones((size, 1)).flatten()
    b = (H @ x).flatten()
    iterSolve = IterativeSolver(H, b)
    alg_x = iterSolve.solve()
    np.testing.assert_allclose(x, alg_x)




