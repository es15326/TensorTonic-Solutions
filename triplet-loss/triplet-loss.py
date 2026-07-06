import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor, positive, negative = np.array(anchor), np.array(positive), np.array(negative)

    if anchor.ndim == 1:
        anchor = np.expand_dims(anchor, axis=0)
        positive = np.expand_dims(positive, axis=0)
        negative = np.expand_dims(negative, axis=0)

    d_pos = np.sum((anchor - positive) ** 2, axis=1)
    d_neg = np.sum((anchor - negative) ** 2, axis=1)

    loss = np.maximum(0, d_pos - d_neg + margin)

    return loss.mean()