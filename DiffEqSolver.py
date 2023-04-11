import math
import matplotlib.pyplot as plt

import numpy as np

from TDMA import TDMAsolver


class DiffEquationSolver():
    def __init__(self, p, q, r, f, a, b, alpha, beta):
        self.p = p
        self.q = q
        self.r = r
        self.f = f
        self.a = a
        self.b = b
        self.cur_grid = []
        self.alpha = alpha
        self.beta = beta

    def create_grid(self, h):
        self.cur_grid = []
        x = self.a
        while not math.isclose(x, self.b):
            self.cur_grid.append(x)
            x = x + h
        self.cur_grid.append(x)

    def calculate_coeffs(self, h):
        # border conditions O(h^2)
        A = []
        B = []
        C = []
        G = []
        # A_0 = 0
        # B_0 = h * self.alpha[0] + 2 * self.alpha[1]
        # C_0 = 2 * self.alpha[1] - h * self.alpha[0]
        # G_0 = -2 * h * self.alpha[2]
        A_0 = 0
        B_0 = -1
        C_0 = 0
        G_0 = self.alpha[2]
        A.append(A_0)
        B.append(B_0)
        C.append(C_0)
        G.append(G_0)

        n = len(self.cur_grid)
        # matrix = np.zeros(shape=(n, n))
        # A_last = 2 * self.beta[1] - h * self.beta[0]
        # B_last = h * self.beta[0] + 2 * self.beta[1]
        # C_last = 0
        # G_last = -2 * h * self.beta[2]
        A_last = 0
        B_last = -1
        C_last = 0
        G_last = self.beta[2]


        # matrix[0, 0] = 1
        #
        # matrix[n-1, n-1] = 1
        # G_0 = self.alpha[2]
        # G.append(G_0)


        for i, x_i in enumerate(self.cur_grid[1:-1], 1):
            # cur_A = -1 * self.p(x_i - h / 2) - self.q(x_i) * h / 2
            # cur_C = -1 * self.p(x_i + h / 2) + self.q(x_i) * h / 2
            # matrix[i, i - 1] = -1.0 / h**2 - self.q(x_i) / (2.0 * h)
            # matrix[i, i] = -2.0 / h**2 + self.r(x_i)
            # matrix[i, i + 1] = 1.0 / h**2 + self.q(x_i) / (2.0 * h)

            cur_A = self.p(x_i) / (h ** 2) - self.q(x_i) / (2 * h)
            cur_C = self.p(x_i) / (h ** 2) + self.q(x_i) / (2 * h)
            cur_B = (2 * self.p(x_i)) / (h ** 2) - self.r(x_i)
            cur_G = self.f(x_i)

            # cur_B = cur_A + cur_C - (h ** 2) * self.r(x_i)
            # cur_A = 1.0 / h**2 - self.q(x_i) / (2.0 * h)
            # cur_B = -2.0 / h**2 + self.r(x_i)
            # cur_C = 1.0 / h**2 + self.q(x_i) / (2.0 * h)
            # cur_G = self.f(x_i)

            A.append(cur_A)
            B.append(cur_B)
            C.append(cur_C)
            G.append(cur_G)

        A.append(A_last)
        B.append(B_last)
        C.append(C_last)
        G.append(G_last)
        # print(matrix)

        return A, B, C, G
        # return matrix, G

    def solve_matrix(self, h):
        A, B, C, G = self.calculate_coeffs(h)
        cur_Y = TDMAsolver(A, B, C, G)
        return cur_Y

    def get_cur_solution(self, h):
        self.create_grid(h)
        y = self.solve_matrix(h)
        return y

    def calculate_delta(self, y_prev, y_next):
        delta = []
        for i in range(0, len(y_next), 2):
            delta.append((y_next[i] - y_prev[i // 2]) / 3)
            if i + 1 < len(y_next):
                delta.append(0)
        for j in range(1, len(delta) - 1, 2):
            delta[j] = (delta[j - 1] + delta[j + 1])/2
        return delta

    def get_res(self, y_prev, y_next):
        delta_x = self.calculate_delta(y_prev, y_next)
        squared_sub = list(map(lambda x: x ** 2, delta_x))
        return np.linalg.norm(delta_x), max(delta_x), delta_x

    def solve(self, epsilon, s):
        R_l_2 = 5
        R_C = 5
        h = abs(self.a - self.b) / 2
        y_next = []
        delta = []
        y_prev = self.get_cur_solution(h)
        plt.plot(self.cur_grid, y_prev, color='r')
        i = 1
        while R_l_2 > epsilon or R_C > epsilon:
            i += 1
            y_next = self.get_cur_solution(h / 2)
            R_l_2, R_C, delta = self.get_res(y_prev, y_next)
            # y_next = y_next + delta
            y_prev = y_next.copy()
            h /= 2
            cur_color = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(self.cur_grid, y_next, color=cur_color)
        y_next = y_next + delta
        plt.grid()
        plt.show()

        return R_l_2, R_C, i
