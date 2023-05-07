import math

import numpy as np
from numpy import typing


class QRDecomposer():
    def __init__(self, A: typing.NDArray):
        self.A = A
        self.Q = np.zeros(A.shape)
        self.R = np.zeros(A.shape)

    def decompose(self):
        n = self.A.shape[0]
        Q = np.identity(n)
        R = self.A.copy()
        for i in range(n):
            for j in range(i + 1, n):
                T = np.identity(n)

                sqrt = math.sqrt(R[i, i] ** 2 + R[j, i] ** 2)
                cos = R[i, i] / sqrt
                sin = -R[j, i] / sqrt

                T[i, i] = cos
                T[j, i] = sin
                T[j, j] = cos
                T[i, j] = -sin

                R = T @ R
                Q = T @ Q
        Q = np.linalg.inv(Q)

        for i in range(n):
            for j in range(i + 1, n):
                R[j, i] = 0

        return Q, R