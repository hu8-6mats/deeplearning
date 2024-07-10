import gate_and
import gate_nand
import gate_or

def XOR(x1: int, x2: int) -> int:
    """
    Computes the XOR logic gate using a combination of NAND, OR, and AND gates.

    Args:
        x1 (int): The first input value, should be either 0 or 1.
        x2 (int): The second input value, should be either 0 or 1.

    Returns:
        int: The output of the XOR gate. Returns 1 if exactly one of the inputs is 1, otherwise returns 0.
    """
    s1 = gate_nand.NAND(x1, x2)
    s2 = gate_or.OR(x1, x2)
    y = gate_and.AND(s1, s2)

    return y