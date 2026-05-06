import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    for r in matrix:
        if isinstance(r, list):
            if len(r) != len(matrix[0]):
                return None
    matrix = np.array(matrix)

    if matrix.size == 0:
        return None

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eig_vals = np.linalg.eigvals(matrix)

    indices = np.lexsort((eig_vals.imag, eig_vals.real))

    return eig_vals[indices]