import numpy as np
import matplotlib.pylab as plt

def relu(x):
    """
    Computes the ReLU (Rectified Linear Unit) function for a given input array.

    The ReLU function is defined as the element-wise maximum of 0 and the input value.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: An array where each element is the result of applying the ReLU function to the corresponding input element.
    """
    return np.maximum(0, x)

if __name__ == '__main__':
    # Generate input values from -5.0 to 5.0 with a step of 0.1
    x = np.arange(-5.0, 5.0, 0.1)

    # Apply the ReLU function to the input values
    y = relu(x)

    # Plot the ReLU function
    plt.plot(x, y, linestyle='-', label='ReLU Function')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ReLU Function')
    plt.legend()
    plt.grid(True)
    plt.show()