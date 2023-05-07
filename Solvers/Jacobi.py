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

class JacobiEigen():
    def __init__(self, A, eps=0.00001):
        self.A = A
        self.eigenvals, self.iterations = self.find_with_Jacobi(eps)

    def find_max_above_diag(self):
        n = self.A.shape[0]
        i_max, j_max = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if i == j: continue
                if abs(self.A[i_max, j_max]) < abs(self.A[i, j]):
                    i_max, j_max = i, j
        return i_max, j_max

    def rotate(self, i, j):
        A = self.A
        n = A.shape[0]
        E = np.identity(n)
        phi = math.atan(2 * A[i, j] / (A[i, i] - A[j, j])) / 2
        c = math.cos(phi)
        s = math.sin(phi)
        E[i, i] = E[j, j] = c
        E[j, i] = s
        E[i, j] = -s

        return E.T @ A @ E

    def find_with_Jacobi(self, eps=0.0000001):
        if not np.allclose(self.A, self.A.T):
            raise ValueError('Матрица должна быть симметричной')

        iterations = 0
        while True:
            iterations += 1
            i, j = self.find_max_above_diag()

            if np.abs(self.A[i, j]) < eps:
                return np.diagonal(self.A).copy(), iterations
            self.A = self.rotate(i, j)

