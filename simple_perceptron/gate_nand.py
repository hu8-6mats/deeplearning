import numpy as np

def NAND(x1: int, x2: int) -> int:
    """
    Computes the NAND logic gate using a perceptron model for two input values.

    Args:
        x1 (int): The first input value.
        x2 (int): The second input value.

    Returns:
        int: The output of the NAND gate. Returns 0 if both inputs are 1, otherwise returns 1.
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b

    if tmp <= 0:
        return 0
    return 1