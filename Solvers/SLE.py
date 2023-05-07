import numpy as np

def build_B_C(A, b):
    n = len(A[0])
    B = np.zeros_like(A)
    C = np.zeros_like(A[0])
    for i in range(n):
        C[i] = b[i] / A[i, i]
        for j in range(n):
            if i == j:
                B[i][j] = 0
            else:
                B[i][j] = -1 * A[i][j] / A[i][i]
    return B, C


def build_B_C_pos_def(A, b):
    N = A.shape[0]
    m = min(A[k, k] - sum(abs(A[k, j]) if k != j else 0 for j in range(N)) for k in range(N))
    m = max(m, 0)
    M = max(A[k, k] + sum(abs(A[k, j]) if k != j else 0 for j in range(N)) for k in range(N))
    alpha = 2 / (m + M)
    B = np.identity(N) - alpha * A
    C = alpha * b
    return B, C


def solve_with_iterating(A, b, eps, verbose=False, pos_definitive=False):
    if pos_definitive:
        B, C = build_B_C_pos_def(A, b)
    else:
        B, C = build_B_C(A, b)

    p = max(abs(np.linalg.eigvals(B)))
    if p >= 1:
        raise ValueError(f'p(B) == {p} >= 1, метод простой итерации не сходится')

    x_prev = np.zeros_like(A[0])
    x_next = B @ x_prev + C
    i = 0
    while np.linalg.norm(x_next - x_prev) > eps:
        i += 1
        x_prev = x_next
        x_next = B @ x_prev + C

    if verbose:
        return x_next, i
    else:
        return x_next


def solve_with_Seidel(A, b, eps, verbose=False, max_iter=1000000):
    n = A.shape[0]
    B, C = build_B_C(A, b)

    x_prev = np.zeros(A[0])
    x_next = np.copy(x_prev)

    k = 0
    while k < max_iter:
        x_next = np.copy(x_prev)
        for i in range(n):
            x_next[i] = sum(B[i, j] * x_next[j] for j in range(i)) + sum(B[i, j] * x[j] for j in range(i + 1, n)) + C[i]
        if np.linalg.norm(x_prev - x_next) < eps / (10 ** n):
            break
        x_prev = x_next

    if verbose:
        return x_next, k
    else:
        return x_next
