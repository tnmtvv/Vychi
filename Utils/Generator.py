import numpy as np


def generate_diag_prevail(n):
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = sum([abs(A[i, j]) for j in range(n)]) + 1
    return A

def generate_symmetric(n):
    a = np.random.rand(n, n)
    return np.tril(a) + np.tril(a, -1).T

def generate_random(n):
    A = np.random.rand(n, n)
    return A

