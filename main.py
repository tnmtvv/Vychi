# This is a sample Python script.
import argparse
import math

import numpy as np
from math import cos

from tabulate import tabulate

import Utils.Generator
from Utils.Plots import *
from Solvers import LU_decompositor
from Solvers.DiffEqSolver import DiffEquationSolver
from Solvers.IterativeSolver import IterativeSolver
from Solvers.BoundaryValSolver import BoundaryProbSolver
from Solvers.MaxEigen import EigenMax
from Utils.Generator import *
from Utils.Utils import *
from Solvers.Jacobi import JacobiEigen
from Solvers.DiffusionEquationSolver import DiffusionSolver

from scipy.linalg import hilbert


def main(num_of_lab):
    if num_of_lab == 1:
        eps = float(input("Enter eps"))

        p = lambda x: -1 / (2 + x)
        q = lambda x: cos(x)
        r = lambda x: 0
        f = lambda x: 1 + x

        #  u(−1) = u(1) = 0.
        alpha = (1, 0, 0)
        beta = (1, 0, 0)

        a = -1
        b = 1

        diffEqSolver = DiffEquationSolver(p, q, r, f, a, b, alpha, beta)
        answ, R_l_2, R_C, i = diffEqSolver.solve(eps, 2)

        print("Сходимость за {} шагов".format(i))
        print(R_l_2)
        print(R_C)

    elif num_of_lab == 4:
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
        answ = decomp.solve(b)
        print(A @ answ)
    elif num_of_lab == 2:
        A = hilbert(3)
        x = np.ones((3, 1)).flatten()
        b = A @ x
        iterSolve = IterativeSolver(A, b)
        answ_iter = iterSolve.solve(eps=0.000001, get_iterations=True)
        print(answ_iter)

        # A_s = hilbert(4)
        # b_s = A_s @ x
        # iterSolve_2 = IterativeSolver(A_s, b_s)
        answ_Seidel = iterSolve.solve_with_Seidel(eps=0.000001, get_iterations=True)
        print(answ_Seidel)
    elif num_of_lab == 3:
        # 1/(2+x)y' + cos(x)y=1+x
        p = lambda x: -1 / (2 + x)
        q = lambda x: cos(x)
        r = lambda x: 0
        f = lambda x: 1 + x

        boundSolve = BoundaryProbSolver(p, q, r, f)
        Utils.Plots.plot(boundSolve, n=10)
    elif num_of_lab == 7:
        size = 20
        eps = 0.0000000001

        A = generate_symmetric(size)
        expected = np.linalg.eigvalsh(A)
        jacEigen = JacobiEigen(A, eps=eps)
        alg, rotations = jacEigen.eigenvals, jacEigen.iterations
        expected.sort()
        alg.sort()

        m, M = gershgorin(A)
        assert (all(m <= eigv and eigv <= M for eigv in alg))

        table = {'Метод Якоби': alg, 'Встроенная функция numpy': expected}
        print(f'Матрица {size}x{size}, eps={eps}, iterations={rotations}')
        print(tabulate(table, headers='keys', tablefmt='psql'))

    elif num_of_lab == 6:
        size = 6
        eps = 0.00000001

        A = generate_random(size)
        eigMax = EigenMax(A)

        eigs, vects = np.linalg.eig(A)
        eig = max(abs(eigs))

        power_eig, power_eigvec, power_iterations = eigMax.power_method(eps)
        dot_prod_eig, dot_prod_eigvec, dot_prod_iterations = eigMax.dot_prod_method(eps)

        power_method = 'Степенной метод'
        dot_prod_method = 'Метод скалярных произведений'

        table = {power_method: [power_eig], dot_prod_method: [dot_prod_eig,], 'Встроенная функция numpy': [eig]}
        print(f'Матрица {size}x{size}, eps={eps}')
        print(f'Число итераций для степенного={power_iterations}')
        print(f'Число итераций для скалярных произведений={dot_prod_iterations}')
        print(tabulate(table, headers='keys', tablefmt='psql'))

        print('Собственный вектор, подобранный степенным методом:')
        print(power_eigvec)
        print('Собственный вектор, подобранный методом скалярных произведений:')
        print(dot_prod_eigvec)

    elif num_of_lab == 9:
        k = 0.001
        max_x = 1
        max_t = 1

        # u = lambda x, t: x ** 3 + t ** 3
        # f = lambda x, t: 3 * t ** 2 - k * 6 * x
        # u = lambda x, t: t * math.sin(x)
        # f = lambda x, t: (1 + k * t) * math.sin(x)
        u = lambda x, t: x ** 2 - t ** 2
        f = lambda x, t: 2 * k - 2 * t
        # f = lambda x, t: 1
        boundary_cond = lambda x, t: x ** 2 - t ** 2

        diffusionSolver = DiffusionSolver(u, f, boundary_cond, 20, 20, max_x, max_t, k)
        plot_diffusion(diffusionSolver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Labs"
    )
    parser.add_argument(
        "num_of_task", type=int, help="Please enter number of the task"
    )
    args = parser.parse_args()
    main(args.num_of_task)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
