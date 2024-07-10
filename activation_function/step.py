import numpy as np
import matplotlib.pylab as plt

def step(x: np.ndarray) -> np.ndarray:
    """
    Computes the step function for a given input array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: An array where each element is 1 if the corresponding input element is greater than 0, otherwise 0.
    """
    return np.array(x > 0, dtype=int)

if __name__ == '__main__':
    # Generate input values from -5.0 to 5.0 with a step of 0.1
    x = np.arange(-5.0, 5.0, 0.1)

    # Apply the step function to the input values
    y = step(x)

    # Plot the step function
    plt.plot(x, y, linestyle='-', label='Step Function')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Step Function')
    plt.legend()
    plt.grid(True)
    plt.show()
