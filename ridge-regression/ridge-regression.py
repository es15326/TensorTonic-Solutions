def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)

    w = np.linalg.inv(X.T.dot(X) + lam * np.eye(X.shape[1])).dot(X.T.dot(y))


    return w