import numpy as np


class IterativeSolver():
    def __init__(self, A, b):
        self.A = A
        self.b = b

    @staticmethod
    def error(x1, x2):
        return np.linalg.norm(x1 - x2)

    def posteriori_error(self, x1, x2, B):
        B_norm = np.linalg.norm(B, ord=2)
        return B_norm / (1 - B_norm) * self.error(x1, x2)

    def transform_pos_definite(self):
        N = self.A.shape[0]
        m = min(self.A[k, k] - sum(abs(self.A[k, j]) if k != j else 0 for j in range(N)) for k in range(N))
        m = max(m, 0)
        M = max(self.A[k, k] + sum(abs(self.A[k, j]) if k != j else 0 for j in range(N)) for k in range(N))
        alpha = 2 / (m + M)
        B = np.identity(N) - alpha * self.A
        C = alpha * self.b
        return B, C

    def transform_diag_prevail(self):
        N = self.A.shape[0]
        B = np.zeros((N, N))
        C = np.zeros((N, 1))
        for i in range(N):
            C[i] = self.b[i] / self.A[i, i]
            for j in range(N):
                B[i, j] = 0 if i == j else -self.A[i, j] / self.A[i, i]

        return B, C

    def solve(self, eps=1.e-6, get_iterations=False, if_pos_definitive=False, max_iter=100000):
        N = self.A.shape[0]
        if if_pos_definitive:
            B, C = self.transform_pos_definite()
        else:
            B, C = self.transform_diag_prevail()

        p = max(abs(np.linalg.eigvals(B)))
        if p >= 1:
            raise ValueError(f'p(B) == {p} >= 1, abs >= 1')

        x = C
        x_next = B @ x + C
        i = 0
        while i < max_iter and self.posteriori_error(x_next, x, B) > eps:
            i += 1
            x = x_next
            x_next = B @ x + C

        if get_iterations:
            print(i)
            return x_next
        else:
            return x_next

    def solve_with_Seidel(self, get_iterations=False, max_iter=100000, eps=1.e-6):
        N = self.A.shape[0]
        B, C = self.transform_diag_prevail()

        x = np.zeros((N, 1))

        k = 0
        while k < max_iter:
            k += 1
            x_next = np.copy(x)
            for i in range(N):
                x_next[i] = sum(B[i, j] * x_next[j] for j in range(i)) + sum(B[i, j] * x[j] for j in range(i + 1, N)) + \
                            C[i]
            if np.linalg.norm(x - x_next) < eps / (10 ** N):
                break
            x = x_next

        if get_iterations:
            return x_next, k
        else:
            return x_next