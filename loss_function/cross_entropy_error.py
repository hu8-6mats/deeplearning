import numpy as np

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    Computes the cross-entropy error between the predicted values and the true values.

    Parameters:
        y(np.ndarray): Predicted values. Each element should be a probability between 0 and 1.
        t(np.ndarray): True values (targets). In the case of one-hot encoding, each row should contain a single 1 with the rest being 0.

    Returns:
        (float): The cross-entropy error between the predicted values and the true values.
    """
    delta = 1e-7  # Small value to prevent log(0)
    return -np.sum(t * np.log(y + delta))

if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    cee = cross_entropy_error(np.array(y), np.array(t))

    print("t =")
    print(t)
    print("y =")
    print(y)
    print("Cross Entropy Error:")
    print(cee)

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    cee = cross_entropy_error(np.array(y), np.array(t))

    print("t =")
    print(t)
    print("y =")
    print(y)
    print("Cross Entropy Error:")
    print(cee)