"""Neural network model."""

from typing import Sequence

import numpy as np
from scipy.special import softmax


class NeuralNetworkAdam:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        self.params["t"] = 0
        for i in range(1, num_layers + 1):
            self.params["m" + str(i)] = np.zeros_like(self.params["W" + str(i)])
            self.params["v" + str(i)] = np.zeros_like(self.params["W" + str(i)])
            self.params["mb" + str(i)] = np.zeros_like(self.params["b" + str(i)])
            self.params["vb" + str(i)] = np.zeros_like(self.params["b" + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.where(X > 0, X, 0.0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        K = -np.max(X)
        return np.exp(X + K) / np.sum(np.exp(X + K), axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        input = X

        # Go through each layer
        for n in range((self.num_layers-1)):
            # Get linear layer output
            input = self.linear(self.params["W" + str(n+1)], input, self.params["b" + str(n+1)])
            self.outputs["L" + str(n+1)] = input

            # Get ReLU activation output
            input = self.relu(input)
            self.outputs["R" + str(n+1)] = input

        # Get final linear layer output
        output = self.linear(self.params["W" + str(self.num_layers)], input, self.params["b" + str(self.num_layers)])
        self.outputs["L" + str(self.num_layers)] = output

        # Compute softmax for cross entropy loss
        self.outputs["S"] = self.softmax(output)

        return output


    def linear_grad(self, input: np.ndarray) -> np.ndarray:
        return np.transpose(input)

    def relu_grad(self, input: np.ndarray) -> np.ndarray:
        output = input > 0
        return output.astype(float)

    def softmax_grad(self, input: np.ndarray, y):
        input[np.arange(input.shape[0]), y] -= 1
        input /= input.shape[0]
        return input

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 0.00001
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        loss = 0.0
        num_samples = X.shape[0]

        # Compute loss
        sq = 0.0
        for n in range(1, self.num_layers+1):
            sq += np.sum(self.params["W" + str(n)] * self.params["W" + str(n)])
        reg_loss = (reg / 2) * sq
        loss = (np.sum(-np.log(self.outputs["S"][np.arange(num_samples), y])) + reg_loss) / num_samples

        # Get timestep
        t = self.params["t"] + 1

        # Upstream gradient is initialized to softmax_grad
        upstream_grad = self.softmax_grad(self.outputs["S"], y)

        # Go through each layer
        for n in range(self.num_layers, 0, -1):
            if n != self.num_layers:
                # ReLU backwards pass
                local_grad = self.relu_grad(self.outputs["L" + str(n)])
                upstream_grad *= local_grad

            if n != 1:
                # Linear backwards pass
                local_grad = self.linear_grad(self.params["W" + str(n)])
                self.gradients["W" + str(n)] = self.outputs["R" + str(n-1)].T @ upstream_grad
                self.gradients["b" + str(n)] = (upstream_grad.T @ np.ones((num_samples,)))

                # Update weights
                # self.params["W" + str(n)] -= lr*(m_hat / (np.sqrt(v_hat) + epsilon))
                # self.params["b" + str(n)] -= lr*(upstream_grad.T @ np.ones((num_samples,)))

                # Gradient update
                upstream_grad = upstream_grad @ local_grad

        # Linear backwards pass
        local_grad = self.linear_grad(self.params["W" + str(1)])
        self.gradients["W" + str(1)] = X.T @ upstream_grad
        self.gradients["b" + str(n)] = (upstream_grad.T @ np.ones((num_samples,)))

        # Calculate Adam
        for n in range(1, self.num_layers+1):
            # W updates
            grad_w = self.gradients["W" + str(n)]
            self.params["m" + str(n)] = beta_1*self.params["m" + str(n)] + (1 - beta_1)*grad_w
            self.params["v" + str(n)] = beta_2*self.params["v" + str(n)] + (1 - beta_2)*(grad_w*grad_w)
            m_hat = self.params["m" + str(n)] / (1 - pow(beta_1, t))
            v_hat = self.params["v" + str(n)] / (1 - pow(beta_2, t))

            # b updates
            grad_b = self.gradients["b" + str(n)]
            self.params["mb" + str(n)] = beta_1*self.params["mb" + str(n)] + (1 - beta_1)*grad_b
            self.params["vb" + str(n)] = beta_2*self.params["vb" + str(n)] + (1 - beta_2)*(grad_b*grad_b)
            mb_hat = self.params["mb" + str(n)] / (1 - pow(beta_1, t))
            vb_hat = self.params["vb" + str(n)] / (1 - pow(beta_2, t))

            # Update weights
            self.params["W" + str(n)] = (1-reg*lr)*self.params["W" + str(n)] - lr*(m_hat / (np.sqrt(v_hat) + epsilon))
            self.params["b" + str(n)] -= lr*(mb_hat / (np.sqrt(vb_hat) + epsilon))

        # Gradient update
        upstream_grad = upstream_grad @ local_grad

        # Update timestep
        self.params["t"] = t

        return loss
