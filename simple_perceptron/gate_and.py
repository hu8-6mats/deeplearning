import numpy as np

def AND(x1: int, x2: int) -> int:
    """
    Computes the AND logic gate for two numpy arrays of inputs element-wise.

    Args:
        x1 (int): The first input array.
        x2 (int): The second input array.

    Returns:
        int: An array with the result of the AND operation applied element-wise.
             Returns 1 where both corresponding elements of x1 and x2 are 1, otherwise returns 0.
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b

    if tmp <= 0:
        return 0
    return 1
