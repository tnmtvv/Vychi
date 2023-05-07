import numpy as np
from numpy import typing


class LUDecomposer():
    def __init__(self, A: typing.NDArray):
        self.A = A
        self.L = np.zeros(A.shape)
        self.U = np.zeros(A.shape)

    def solve(self, b):
        answ = self.Gauss_U(self.Gauss_L(b))
        # answ = list(map(lambda x: round(x), answ))
        return answ
        # return answ

    def decompose(self):
        n = np.shape(self.A)[1]
        for i in range(n):
            self.L[i, 0] = self.A[i, 0]
            self.U[0, i] = self.A[0, i] / self.L[0, 0]
        for j in range(1, n):
            self.U[j, j] = 1
        # calculate the i column of L, then calculate the i row of U
        for i in range(1, n):
            for j in range(i, n):
                sum = 0
                for k in range(0, i):
                    sum += self.L[j, k] * self.U[k, i]
                self.L[j, i] = self.A[j, i] - sum
            for j in range(i + 1, n):
                sum = 0
                for k in range(0, i):
                    sum += self.L[i, k] * self.U[k, j]
                self.U[i, j] = (self.A[i, j] - sum) / self.L[i, i]
        return self.L, self.U

    def Gauss_L(self, b):
        n = np.shape(self.L)[1]
        y = []
        y.append(b[0] / self.L[0][0])
        for i in range(1, n):
            sum = 0
            for j in range(0, i):
                sum += self.L[i][j] * y[j]
            y.append((b[i] - sum)/self.L[i][i])
        return y
    
    def Gauss_U(self, y):
        n = np.shape(self.L)[1]
        x = []
        x.append(y[-1])
        for i in range(2, n + 1):
            sum = 0
            for j in range(1, i):
                sum += self.U[-i][-j] * x[j - 1]
            cur_x = y[-i] - sum
            x.append(cur_x)
        return list(reversed(x))

