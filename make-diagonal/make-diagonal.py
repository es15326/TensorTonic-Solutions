import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    D = np.zeros((len(v), len(v)), dtype=np.float64)

    for i, v in enumerate(v):
        D[i, i] = v

    return D
