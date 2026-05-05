import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a, b = np.array(a), np.array(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if not norm_a or not norm_b:
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)