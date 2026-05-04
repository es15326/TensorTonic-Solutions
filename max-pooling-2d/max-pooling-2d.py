def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    h = len(X)
    w = len(X[0])

    out_h = h // pool_size
    out_w = w // pool_size

    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]

    for i in range(out_h):
        for j in range(out_w):

            start_i = i * pool_size
            start_j = j * pool_size

            curr_max =  float('-inf')
            for m in range(pool_size):
                for n in range(pool_size):
                    if X[start_i + m][start_j + n] > curr_max:
                        curr_max = X[start_i + m][start_j + n]

            out[i][j] = curr_max

    return out