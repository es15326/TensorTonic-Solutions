import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len, dtype=np.float64).reshape(seq_len, 1)
    i = np.arange(d_model, dtype=np.float64).reshape(1, d_model)
    
    angle_den = base ** ((2 * (i // 2)) / d_model)

    angles = pos / angle_den

    pos_enc = np.zeros((seq_len, d_model), dtype=np.float64)

    pos_enc[:, 0::2] = np.sin(angles[:, 0::2])
    pos_enc[:, 1::2] = np.cos(angles[:, 1::2])

    return pos_enc