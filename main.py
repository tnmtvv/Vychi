# This is a sample Python script.
import math

import numpy as np

import LU_decompositor
from DiffEqSolver import DiffEquationSolver

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
def main():
    # # p = lambda x: -1 / (x - 3)
    # # q = lambda x: 1 + (x / 2)
    # # r = lambda x: -1 * math.exp(x / 2)
    # # f = lambda x: 2 - x
    #
    # p = lambda x: -1 / (2 + x)
    # q = lambda x: math.cos(x)
    # r = lambda x: 0
    # f = lambda x: 1 + x
    #
    # # p = lambda x: -1 / (x - 3)
    # # q = lambda x: -x
    # # r = lambda x: math.log(2 + x, math.e)
    # # f = lambda x: 1 - x / 2
    #
    # q_new = lambda x: q(x) / p(x)
    # r_new = lambda x: r(x) / p(x)
    # f_new = lambda x: f(x) / p(x)
    #
    # #  u(−1) = u(1) = 0.
    # alpha = (1, 0, 0)
    # beta = (1, 0, 0)
    #
    # # u'(−1) = u'(1) + 1/2 u(1) = 0.
    # # alpha = (0, 1, 0)
    # # beta = (0.5, 1, 0)
    #
    # a = -1
    # b = 1
    #
    # diffEqSolver = DiffEquationSolver(p, q, r, f, a, b, alpha, beta, )
    # R_l_2, R_C, i = diffEqSolver.solve(0.000001, 2)
    #
    # print("Сходимость за {} шагов".format(i))
    # print(R_l_2)
    # print(R_C)

    A = np.asarray([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    b = [2, 3, 4, 5]
    cond_num_A = np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A))
    decomp = LU_decompositor.LUDecomposer(A)
    L, U = decomp.decompose()
    # if_worked = np.allclose(A - L @ U, np.zeros((4, 4)))
    print(L @ U)
    cond_num_L = np.linalg.norm(L) * np.linalg.norm(np.linalg.inv(L))
    cond_num_U = np.linalg.norm(U) * np.linalg.norm(np.linalg.inv(U))
    print('Число обусловленности A = {}'.format(cond_num_A))
    print('Число обусловленности L = {}'.format(cond_num_L))
    print('Число обусловленности U = {}'.format(cond_num_U))
    answ = decomp.Gauss_U(decomp.Gauss_L(b))
    print(A @ answ)

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
