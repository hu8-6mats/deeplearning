# deeplearning
Repository for DeepLearning learning

See below for the Japanese version of the README.

[README_jp.md](README_jp.md)

## 0. Setup Instructions
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. File Descriptions

### neural_network/dataset/mnist.py

Defines functions to load the MNIST dataset.

Use the load_mnist function to load MNIST data.  
Setting normalize=True normalizes the image data, and one_hot_label=True loads labels in one-hot encoding.

### neural_network/two_layer_net.py

Defines the TwoLayerNet class.

This class constructs a 2-layer neural network, performs training,  inference, and gradient computation.  
The __init__ method initializes the network structure, setting weights and biases.  
The predict method infers input data, the loss method computes the loss (cross-entropy error).  
The accuracy method evaluates accuracy, and numerical_gradient and gradient compute gradients.

### neural_network/train_two_layer_net.py

Contains the neural network training process.

Loads the MNIST dataset, initializes the network using the TwoLayerNet class.  
Trains the network for a specified number of iterations using mini-batches, displaying intermediate progress and final accuracy.

## 2. Usage

```zsh
cd neural_network
python3 train_two_layer_net.py
Running train_two_layer_net.py initiates training of a neural network using TwoLayerNet.
```

Displays loss per iteration, evaluates training and test data accuracy periodically.

Final accuracy on the test data is shown, evaluating the network's performance.

