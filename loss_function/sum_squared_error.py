import numpy as np

def sum_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    Computes the sum of squared error between the predicted values and the true values.

    Parameters:
        y(np.ndarray): Predicted values.
        t(np.ndarray): True values (targets).

    Returns:
        (float): The sum of squared error between the predicted values and the true values.
    """
    return 0.5 * np.sum((y - t) ** 2)

if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    sse = sum_squared_error(np.array(y), np.array(t))

    print("t =")
    print(t)
    print("y =")
    print(y)
    print("Sum Squared Error:")
    print(sse)

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    sse = sum_squared_error(np.array(y), np.array(t))

    print("t =")
    print(t)
    print("y =")
    print(y)
    print("Sum Squared Error:")
    print(sse)