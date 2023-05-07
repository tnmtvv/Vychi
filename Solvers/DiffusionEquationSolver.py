import seaborn as sns
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import hilbert
import unittest
from parameterized import parameterized, parameterized_class
from tabulate import tabulate
import numpy.testing
import matplotlib as mpl
import functools
import scipy as sp
import scipy.special
import scipy.misc
import scipy.integrate
from matplotlib import cm
from colorama import Fore
from Utils.Utils import *

# u_t(x, t) = κu_xx (x, t) + f (x, t)

class DiffusionSolver():
    def __init__(self, u, f, boundary_func, N_x, N_t, a_x, b_t, k):
        self.x_grid = grid(0, a_x, N_x + 1)
        self.t_grid = grid(0, b_t, N_t + 1)
        self.u = u
        self.f = f
        self.L = np.zeros((N_x + 1, N_t + 1))
        self.tau = b_t / N_t
        self.h = a_x / N_x
        self.k = k
        self.N_t = N_t
        self.N_x = N_x
        self.b_t = b_t
        self.boundary_func = boundary_func
        self.start = boundary_func(self.x_grid[0], self.t_grid[0])
        self.end = boundary_func(self.x_grid[-1], self.t_grid[0])

        for i in range(N_x + 1):
            self.L[i, 0] = self.boundary_func(self.x_grid[i], self.t_grid[0])
        for i in range(N_t + 1):
            self.L[0, i] = boundary_func(self.x_grid[0], self.t_grid[i])
            self.L[N_x, i] = boundary_func(self.x_grid[N_x], self.t_grid[i])

    def reset_L(self):
        N_x = self.N_x
        N_t = self.N_t
        u = self.u
        self.L = np.zeros((N_x + 1, N_t + 1))

        for i in range(N_x + 1):
            self.L[i, 0] = self.boundary_func(self.x_grid[i], self.t_grid[0])
        for i in range(N_t + 1):
            self.L[0, i] = self.boundary_func(self.x_grid[0], self.t_grid[i])
            self.L[N_x, i] = self.boundary_func(self.x_grid[N_x], self.t_grid[i])

    def explicit_scheme(self):
        # self.reset_L()
        L = self.L
        if not 2 * self.k * self.tau <= self.h ** 2:
            print(Fore.RED + 'Явная схема неустойчива!')

        for t in range(1, self.N_t + 1):
            for x in range(1, self.N_x):
                diff = L[x - 1, t - 1] - 2 * L[x, t - 1] + L[x + 1, t - 1]
                L[x, t] = L[x, t - 1] + self.tau * (self.k / self.h ** 2 * diff + self.f(self.x_grid[x], self.t_grid[t - 1]))

        else:
            return L

    def implicit_scheme(self):
        self.reset_L()
        for t in range(1, self.N_t + 1):
            lhs = np.zeros((self.N_x - 1, self.N_x-1))
            rhs = np.zeros(self.N_x-1)

            # lhs[0, 0] = -(self.tau * self.k / self.h + 1)
            # lhs[0, 1] = self.tau * self.k / self.h
            # rhs[0] = -self.L[0, t - 1] - self.tau * self.f(self.x_grid[0], self.t_grid[t])
            #
            # lhs[self.N_x, self.N_x] = self.tau * self.k / self.h - 1
            # lhs[self.N_x, self.N_x - 1] = -self.tau * self.k / self.h
            # rhs[self.N_x] = -self.L[self.N_x, t - 1] - self.tau * self.f(self.x_grid[self.N_x], self.t_grid[t])

            coef = self.h ** 2 / (self.tau * self.k )
            for x in range(0, self.N_x - 1):
                lhs[x, x] = -2 - coef
                rhs[x] = - coef * self.L[x, t - 1] - self.tau * coef * self.f(self.x_grid[x], self.t_grid[t]) - self.L[x, t]

            rhs[self.N_x - 2] = rhs[self.N_x - 2] - self.L[-1, t]

            for x in range(1, self.N_x - 2):
                lhs[x, x - 1] = lhs[x, x + 1] = 1
            lhs[0, 1] = 1
            lhs[self.N_x - 2, self.N_x - 3] = 1

            self.L[1:-1, t] = np.linalg.solve(lhs, rhs)

        else:
            return self.L

    def implicit_scheme_2(self):
        self.reset_L()
        alpha = self.tau * self.k / self.h ** 2
        for t in range(1, self.N_t + 1):
            lhs = np.zeros((self.N_x + 1, self.N_x + 1))
            rhs = np.zeros(self.N_x + 1)

            lhs[0, 0] = 2 - 2 * alpha
            lhs[0, 1] = alpha
            rhs[0] = 0

            lhs[self.N_x, self.N_x] = 2 - 2 * alpha
            lhs[self.N_x, self.N_x - 1] = alpha
            rhs[self.N_x] = -self.L[self.N_x, t - 1] - self.tau * self.f(self.x_grid[self.N_x], self.t_grid[t])

            for x in range(1, self.N_x):
                lhs[x, x] = 2 + 2 * alpha
                lhs[x, x - 1] = lhs[x, x + 1] = -alpha

                rhs[x] = alpha * self.L[x - 1, t - 1] - 2 * alpha * self.L[x, t - 1] + alpha * self.L[x + 1, t - 1] +\
                         self.tau * self.f(self.x_grid[x], self.t_grid[t] + self.tau/2)

            self.L[:, t] = np.linalg.solve(lhs, rhs)

        else:
            return self.L