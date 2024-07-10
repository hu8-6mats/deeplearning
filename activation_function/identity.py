import numpy as np
import matplotlib.pylab as plt

def identity(x: np.ndarray) -> np.ndarray:
    """
    Identity function that returns the input array unchanged.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The input array unchanged.
    """
    return x

if __name__ == '__main__':
    # Generate input values from -5.0 to 5.0 with a step of 0.1
    x = np.arange(-5.0, 5.0, 0.1)

    # Apply the identity function to the input values
    y = identity(x)

    # Plot the identity function
    plt.plot(x, y, linestyle='-', label='Identity Function')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Identity Function')
    plt.legend()
    plt.grid(True)
    plt.show()