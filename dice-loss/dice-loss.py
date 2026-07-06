import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p, y = np.array(p), np.array(y)

    if p.ndim > 1:
        p = p.flatten()
        y = y.flatten()

    intersection = np.sum(p * y)

    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    loss = 1 - dice

    return loss.mean()