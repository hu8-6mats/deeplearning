import sys, os
from typing import Dict

sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        """
        Initialize the two-layer neural network.

        Parameters:
        - input_size (int): The size of the input layer.
        - hidden_size (int): The size of the hidden layer.
        - output_size (int): The size of the output layer.
        - weight_init_std (float): The standard deviation for weight initialization (default is 0.01).
        """
        # Initialize weights and biases
        self.params: Dict[str, np.ndarray] = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # Generate layers
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Parameters:
        - x (np.ndarray): The input data.

        Returns:
        - np.ndarray: The predicted output.
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Compute the loss (cross-entropy error) for the given input and true labels.

        Parameters:
        - x (np.ndarray): The input data.
        - t (np.ndarray): The true labels.

        Returns:
        - float: The loss value.
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Compute the accuracy for the given input and true labels.

        Parameters:
        - x (np.ndarray): The input data.
        - t (np.ndarray): The true labels.

        Returns:
        - float: The accuracy value.
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute the numerical gradient for the network parameters.

        Parameters:
        - x (np.ndarray): The input data.
        - t (np.ndarray): The true labels.

        Returns:
        - Dict[str, np.ndarray]: A dictionary containing the gradients of the weights and biases.
        """
        loss_W = lambda W: self.loss(x, t)
        grads: Dict[str, np.ndarray] = {}

        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute the gradients for the network parameters using backpropagation.

        Parameters:
        - x (np.ndarray): The input data.
        - t (np.ndarray): The true labels.

        Returns:
        - Dict[str, np.ndarray]: A dictionary containing the gradients of the weights and biases.
        """
        # Forward pass
        self.loss(x, t)

        # Backward pass
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # Setting gradients
        grads: Dict[str, np.ndarray] = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads


if __name__ == "__main__":
    # サンプルデータの生成
    np.random.seed(0)
    x = np.random.rand(100, 784)  # 入力データ
    t = np.random.rand(100, 10)  # ラベルデータ

    # ハイパーパラメータ
    input_size = 784
    hidden_size = 50
    output_size = 10

    # ネットワークの初期化
    network = TwoLayerNet(input_size, hidden_size, output_size)

    # ハイパーパラメータ
    iters_num = 10000
    train_size = x.shape[0]
    batch_size = 100
    learning_rate = 0.1

    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x[batch_mask]
        t_batch = t[batch_mask]

        # 勾配の計算
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        # 学習過程の表示
        loss = network.loss(x_batch, t_batch)
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    # 精度の表示
    accuracy = network.accuracy(x, t)
    print(f"Final Accuracy: {accuracy}")
