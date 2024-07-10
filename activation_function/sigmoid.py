import numpy as np
import matplotlib.pylab as plt

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function for a given input array.

    The sigmoid function is defined as 1 / (1 + exp(-x)), where exp is the exponential function.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: An array where each element is the result of applying the sigmoid function to the corresponding input element.
    """
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    # Generate input values from -5.0 to 5.0 with a step of 0.1
    x = np.arange(-5.0, 5.0, 0.1)

    # Apply the sigmoid function to the input values
    y = sigmoid(x)

    # Plot the sigmoid function
    plt.plot(x, y, linestyle='-', label='Sigmoid Function')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sigmoid Function')
    plt.legend()
    plt.grid(True)
    plt.show()