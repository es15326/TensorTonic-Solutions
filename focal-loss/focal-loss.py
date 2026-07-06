import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p, y = np.array(p), np.array(y)
    p = np.clip(p, 1e-15, 1 - 1e-15)

    term1 = (1 - p) ** gamma * y * np.log(p)
    term2 = p ** gamma * (1 - y) * np.log(1 - p)

    loss = -(term1 + term2)

    return loss.mean()