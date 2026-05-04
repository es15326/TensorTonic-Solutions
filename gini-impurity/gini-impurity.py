import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    def calculate_gini(y):
        if not y:
            return 0.0

        _, counts = np.unique(y, return_counts=True)

        probs = counts / len(y)

        return 1 - np.sum(probs ** 2)

    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)

    N_l, N_r = len(y_left), len(y_right)
    N = N_l + N_r

    if not N:
        return 0.0

    weighted_gini = (N_l / N) * gini_left + (N_r / N) * gini_right

    return float(weighted_gini)