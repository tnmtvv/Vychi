import numpy as np

def gershgorin(A):
    n = A.shape[0]
    m = min(A[k, k] - sum(abs(A[k, j]) if k != j else 0 for j in range(n)) for k in range(n))
    M = max(A[k, k] + sum(abs(A[k, j]) if k != j else 0 for j in range(n)) for k in range(n))
    return m, M

def dot(a, b):
        return np.dot(a.T, b)[0]

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def grid(a, b, N):
    return np.linspace(a, b, N)