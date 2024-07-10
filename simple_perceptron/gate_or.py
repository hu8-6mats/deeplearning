import numpy as np

def OR(x1: int, x2: int) -> int:
    """
    Computes the OR logic gate using a perceptron model for two input values.

    Args:
        x1 (int): The first input value.
        x2 (int): The second input value.

    Returns:
        int: The output of the OR gate. Returns 1 if at least one of the inputs is 1, otherwise returns 0.
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b

    if tmp <= 0:
        return 0
    return 1
