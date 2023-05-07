import seaborn as sns
import numpy as np
import math
from Utils.Utils import *
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import hilbert
import unittest
from parameterized import parameterized, parameterized_class
from tabulate import tabulate
import numpy.testing
from scipy.linalg import eigh


class EigenMax():
    def __init__(self, A):
        self.A = A

    def power_method(self, eps, max_iterations=1000):
        N = self.A.shape[0]
        x = np.random.rand(N, 1)
        eig = 0
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            x_next = self.A @ x
            eig_next = math.sqrt(dot(x_next, x_next) / dot(x, x))

            if np.linalg.norm(x_next) > 1000:
                x_next = normalize(x_next)

            if abs(eig - eig_next) < eps:
                return eig_next, x_next, iterations

            x = x_next
            eig = eig_next

        return eig_next, x_next, iterations

    def dot_prod_method(self, eps, max_iterations=10000):
        N = self.A.shape[0]
        x = np.random.rand(N, 1)
        y = x
        eig = 0
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            x_next = self.A @ x
            y_next = self.A.T @ y
            eig_next = dot(x_next, y_next) / dot(x, y_next)

            if np.linalg.norm(x_next) > 1000:
                x_next = normalize(x_next)
            if np.linalg.norm(y_next) > 1000:
                y_next = normalize(y_next)

            if abs(eig - eig_next) < eps:
                return eig_next, x_next, iterations

            x = x_next
            eig = eig_next

        return eig_next, x_next, iterations

