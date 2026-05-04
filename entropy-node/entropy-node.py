import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if not y:
        return 0.0
        
    value, counts = np.unique(y, return_counts=True)

    probs = counts / len(y)

    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])

    return float(entropy)