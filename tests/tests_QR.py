import numpy as np
import pytest
from scipy.linalg import hilbert
from Solvers.QR_decomposer import QRDecomposer

@pytest.mark.parametrize("size", [i for i in range(2, 10)])
def test_decomp_on_Hilbert(size):
    H = hilbert(size)
    qr = QRDecomposer(H)
    Q, R = qr.decompose()
    result = Q @ R
    if size == 3:
        print()
        print(Q)
        print(R)
    np.testing.assert_allclose(H, result)
