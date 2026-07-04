import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    if x.ndim == 1:
        return np.exp(x - max(x)) / sum(np.exp(x - max(x)))

    shift = np.max(x, axis=1, keepdims=True)
    x -= shift
    soft_max = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return soft_max
                                                      

                                                      