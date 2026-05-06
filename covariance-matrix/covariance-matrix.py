import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    N = len(X)

    if X.ndim != 2 or N < 2:
        return None

    X_centered = X - X.mean(axis=0)

    cov_ = (1 / (N - 1)) * X_centered.T.dot(X_centered)

    return cov_