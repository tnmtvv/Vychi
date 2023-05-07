import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot(boundSolve, n, start_from=2, expected=None):
    fix, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
    g = np.linspace(boundSolve.interval[0], boundSolve.interval[1], 100)
    solutions_ax = ax[0]
    solutions_ax.title.set_text('Найденные решения дифура для различных N')
    solutions_ax.set_xlabel('x', fontsize=20)
    solutions_ax.set_ylabel('y', fontsize=20)
    for n in range(start_from, n + 1):
        actual = boundSolve.galerkin(n)
        solutions_ax.plot(g, actual(g), label=f'N={n}')
    solutions_ax.legend(prop={'size': 13})

    expected_ax = ax[1]
    if expected != None:
        expected = np.vectorize(expected)
        expected_ax.plot(g, expected(g), label=f'Точное решение')
    expected_ax.title.set_text(f'Сравнение точного решения и решения найденного для N={n}')
    expected_ax.set_xlabel('x', fontsize=20)
    expected_ax.set_ylabel('y', fontsize=20)
    expected_ax.plot(g, actual(g), label=f'Решение для N={n}')
    expected_ax.legend(prop={'size': 13})
    plt.show()


def plot_for(ax, X, T, Z, title):
    ax.plot_surface(X, T, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('t', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)


def plot_diffusion(diffSolver):
    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(30, 10), dpi=80)

    x_grid = diffSolver.x_grid
    t_grid = diffSolver.t_grid

    L_explicit = diffSolver.explicit_scheme()
    L_implicit = diffSolver.implicit_scheme()
    X, T = np.meshgrid(x_grid, t_grid)
    explicit_ax = ax[0]
    implicit_ax = ax[1]

    plot_for(explicit_ax, X, T, L_explicit, 'Явная схема')
    plot_for(implicit_ax, X, T, L_implicit, 'Неявная схема')
    plt.show()