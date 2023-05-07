import scipy as sp
import numpy as np
import math
from math import sin, exp, sqrt


class BoundaryProbSolver():
    def __init__(self, p, q, r, f, expected=None, interval=(-1,1)):
        self.p = p
        self.q = q
        self.r = r
        self.f = f
        self.expected = expected
        self.interval = interval

    @staticmethod
    def jacobi(i):
        return lambda x: (1 - x**2) * sp.special.eval_jacobi(i, 1, 1, x)

    @staticmethod
    def df(func, ord=1):
        return lambda x0: sp.misc.derivative(func, x0, n=ord, dx=1e-2)

    def dot(self, f, g):
        integrand = lambda x: f(x) * g(x)
        return sp.integrate.quad(integrand, self.interval[0], self.interval[1])[0]

    def galerkin(self, N):
        L = lambda w: lambda x: self.p(x) * BoundaryProbSolver.df(w, 2)(x) + self.q(x) * BoundaryProbSolver.df(w)(x) + self.r(x) * w(x)
        L = np.vectorize(L)
        # набор базисных функций -- многочленов якоби, w
        w = [BoundaryProbSolver.jacobi(i) for i in range(N)]
        Lw = L(w)
        lhs = np.zeros((N, N))
        rhs = np.zeros((N, 1))

        for i in range(N):
            for j in range(N):
                lhs[i, j] = self.dot(Lw[j], w[i])
            rhs[i] = self.dot(self.f, w[i])

        c = np.linalg.solve(lhs, rhs)
        return lambda x: sum(c[i] * w[i](x) for i in range(N))