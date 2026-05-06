import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    trace_ = 0
    for r in range(len(A)):
        for c in range(len(A[0])):
            if r == c:
                trace_ += A[r][c]

    return trace_
