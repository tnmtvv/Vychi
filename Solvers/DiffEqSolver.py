import math
import matplotlib.pyplot as plt

import numpy as np

from Solvers.TDMA import TDMAsolver


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
        self.errors = []

    def create_grid(self, h):
        # self.cur_grid = np.linspace(self.a, self.b, n)
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

        A_0 = 0
        B_0 = 1
        C_0 = 0
        G_0 = self.alpha[2]
        A.append(A_0)
        B.append(B_0)
        C.append(C_0)
        G.append(G_0)

        for i, x_i in enumerate(self.cur_grid[1:-1], 1):
            cur_A = self.p(x_i) / (h ** 2) - self.q(x_i) / (2 * h)
            cur_C = self.p(x_i) / (h ** 2) + self.q(x_i) / (2 * h)
            cur_B = (2 * self.p(x_i)) / (h ** 2) - self.r(x_i)
            cur_G = self.f(x_i)

            A.append(cur_A)
            B.append(cur_B)
            C.append(cur_C)
            G.append(cur_G)

        A_last = 0
        B_last = 1
        C_last = 0
        G_last = self.beta[2]

        A.append(A_last)
        B.append(B_last)
        C.append(C_last)
        G.append(G_last)

        return A, B, C, G

    def solve_matrix(self, h):
        A, B, C, G = self.calculate_coeffs(h)
        cur_Y = TDMAsolver(A, B, C, G)
        return cur_Y

    def get_cur_solution(self, h, n):
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
        R_C = np.linalg.norm(delta_x, ord=2) / math.sqrt(len((y_next)))
        squared_sub = list(map(lambda x: x ** 2, delta_x))
        return R_C, max(delta_x), delta_x

    def solve(self, epsilon, draw_plot=False):
        R_l_2 = 5
        R_C = 5
        h = abs(self.a - self.b) / 2
        n = 2
        y_next = []
        delta = []
        y_prev = self.get_cur_solution(h, n)
        i = 1
        plt.plot(self.cur_grid, y_prev, color='g')
        while R_l_2 > epsilon:
            self.errors.append(R_l_2)
            i += 1
            y_next = self.get_cur_solution(h / 2, n * 2)
            R_l_2, R_C, delta = self.get_res(y_prev, y_next)
            y_prev = y_next.copy()
            h /= 2
            n *= 2
            cur_color = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(self.cur_grid, y_next, color=cur_color)
        cur_color = (1, 0, 0)

        plt.plot(self.cur_grid, y_next, color=cur_color)
        y_next = y_next + delta
        plt.grid()
        plt.show()
        # if draw_plot:
        #     self.plot_solution(i, y_next)

        return y_next, R_l_2, R_C, i

    # def solve(self, epsilon, s):
    #     R_l_2 = 5
    #     R_C = 5
    #     h = abs(self.a - self.b) / 2
    #     y_next = []
    #     delta = []
    #     y_prev = self.get_cur_solution(h, 2)
    #     plt.plot(self.cur_grid, y_prev, color='r')
    #     i = 1
    #     while R_l_2 > epsilon:
    #         i += 1
    #         y_next = self.get_cur_solution(h / 2, 2 ** i)
    #         R_l_2, R_C, delta = self.get_res(y_prev, y_next)
    #         # y_next = y_next + delta
    #         y_prev = y_next.copy()
    #         h /= 2
    #         cur_color = (np.random.random(), np.random.random(), np.random.random())
    #         plt.plot(self.cur_grid, y_next, color=cur_color)
    #     y_next = y_next + delta
    #     plt.grid()
    #     plt.show()
    #     return y_next, R_l_2, R_C, i

    def plot_solution(self, it_num, answ, true_func=None):
        num_of_intervals = [pow(2, i) for i in range(1, it_num)]
        fig, ax = plt.subplots(1, 2, figsize=(30, 10), dpi=80)
        solution_ax = ax[0]
        error_ax = ax[1]
        g = self.cur_grid
        solution_ax.title.set_text('Решение:')
        solution_ax.set_xlabel('x', fontsize=20)
        solution_ax.set_ylabel('y', fontsize=20)
        solution_ax.plot(g, answ, label='Найденное решение')

        if true_func:
            solution_ax.plot(g, true_func(), label='Точное решение')
        solution_ax.legend(prop={'size': 20})

        error_ax.title.set_text('Зависимость ошибки от количества узлов сетки')
        error_ax.set_yscale('log')
        error_ax.set_xscale('log', basex=2)
        error_ax.set_xlabel('Количество узлов', fontsize=20)
        error_ax.set_ylabel('Оценка точности ответа', fontsize=20)
        error_ax.plot(num_of_intervals, self.errors)

