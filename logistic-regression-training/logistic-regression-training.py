import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    y = y.reshape(-1, 1)

    n_samples, n_features = X.shape

    w = np.zeros((n_features, 1))
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        predictions = _sigmoid(z)

        error = predictions - y

        dl_dw = (1 / n_samples) * np.dot(X.T, error)
        dl_db = (1 / n_samples) * np.sum(error)

        w -= dl_dw * lr
        b -= dl_db * lr

    return (w.flatten(), b)