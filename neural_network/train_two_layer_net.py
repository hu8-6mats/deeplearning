import numpy as np
from typing import Tuple, List
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# Load data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Initialize network
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list: List[float] = []
train_acc_list: List[float] = []
test_acc_list: List[float] = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # Mini-batch creation
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Compute gradients using backpropagation
    grad = network.gradient(x_batch, t_batch)

    # Update parameters
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # Compute loss and append to the training loss list
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # Calculate accuracy periodically
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")
