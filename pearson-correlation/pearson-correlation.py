import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.array(X)

    X_centered = X - np.mean(X, axis=0)
    N = len(X)
    std_ = np.std(X_centered, axis=0, ddof=1)

    safe_std = np.where(std_ == 0, 1, std_)

    cov_ = (1 / (N - 1)) * X_centered.T.dot(X_centered)

    R = cov_ / np.outer(safe_std, safe_std)

    np.fill_diagonal(R, 1.0)

    zero_std_mask = (std_ == 0)
    if np.any(zero_std_mask):
        mask_2d = zero_std_mask[:, None] | zero_std_mask[None, :]
        R[mask_2d] = np.nan
        

    return np.clip(R, -1.0, 1.0)