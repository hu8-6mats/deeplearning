import numpy as np
import matplotlib.pylab as plt

import numpy as np

def softmax(a: np.ndarray) -> np.ndarray:
    """
    Computes the softmax function for a given input array.

    The softmax function takes an array of numbers as input and outputs an array
    of the same shape where each element is the result of applying the softmax
    function to the corresponding input element.

    Args:
        a (np.ndarray): The input array.

    Returns:
        np.ndarray: An array where each element is the result of applying the softmax
                    function to the corresponding input element.
    """
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

if __name__ == '__main__':
    # Generate input values from -5.0 to 5.0 with a step of 0.1
    x = np.arange(-5.0, 5.0, 0.1)

    # Apply the softmax function to the input values
    y = softmax(x)

    # Plot the softmax function
    plt.plot(x, y, linestyle='-', label='Softmax Function')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Softmax Function')
    plt.legend()
    plt.grid(True)
    plt.show()