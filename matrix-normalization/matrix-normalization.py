import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    try:
        matrix = np.array(matrix, dtype=np.float64)

        if matrix.ndim != 2:
            return None

        if norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norm = np.sqrt(np.sum(np.square(matrix), axis=axis, keepdims=True))
        elif norm_type == 'max':
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None

        norm = np.where(norm == 0, 1, norm)

        return matrix / norm
    except (ValueError, TypeError, AttributeError):
        return None