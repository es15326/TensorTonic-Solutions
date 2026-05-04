def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here
    h = len(X)
    w = len(X[0])

    out_h = h // pool_size
    out_w = w // pool_size

    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]

    for r in range(out_h):
        for c in range(out_w):
            start_r = r * pool_size
            start_c = c * pool_size

            curr_sum = 0.0
            for m in range(pool_size):
                for n in range(pool_size):
                    curr_sum += X[start_r + m][start_c + n]

            out[r][c] = curr_sum / (pool_size * pool_size)

    return out