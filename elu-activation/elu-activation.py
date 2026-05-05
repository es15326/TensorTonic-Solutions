import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    elu = []

    for elm in x:
        if elm > 0:
            elu.append(elm)
        else:
            elu.append(alpha * (math.exp(elm) - 1))

    return elu
    