import numpy as np

# Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(A, B, C, G):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''

    # M = [[0] * 3 for _ in range(len(A))]
    # M[0][0] = C[0] / B[0]
    # M[0][1] = -G[0] / B[0]
    # for i in range(1, len(A)):
    #     M[i][0] = C[i] / (B[i] - A[i] * M[i - 1][0])
    #     M[i][1] = (A[i] * M[i - 1][1] - G[i]) / (B[i] - A[i] * M[i - 1][0])
    # M[-1][2] = M[-1][1]
    # for i in range(len(A) - 2, -1, -1):
    #     M[i][2] = M[i][0] * M[i + 1][2] + M[i][1]
    # return np.asarray(M)[:, 2]

    s = []
    t = []
    answ = []

    s.append(C[0] / B[0])
    t.append(-1 * G[0] / B[0])

    for i, c in enumerate(C[1:], 1):
        s.append(c/(B[i] - A[i] * s[i-1]))
        t.append((A[i] * t[i-1] - G[i]) / (B[i] - A[i] * s[i-1]))

    for i in range(len(G)):
        if i == 0:
            answ.append(t[-1])
        else:
            answ.append(s[-(i + 1)] * answ[i - 1] + t[-(i + 1)])
    return list(reversed(answ))
    # nf = len(d)  # number of equations
    # ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    # for it in range(1, nf):
    #     mc = ac[it - 1] / bc[it - 1]
    #     bc[it] = bc[it] - mc * cc[it - 1]
    #     dc[it] = dc[it] - mc * dc[it - 1]
    #
    # xc = bc
    # xc[-1] = dc[-1] / bc[-1]
    #
    # for il in range(nf - 2, -1, -1):
    #     xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    # return p