import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a, b, y = np.array(a), np.array(b), np.array(y)

    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]

    d = np.linalg.norm(a - b, axis=1)
    loss = y * np.square(d) + (1 - y) * np.square(np.maximum(0, margin - d))

    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)