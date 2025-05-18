import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        # Weight initialization (small values)
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((output_dim, 1))

        # Placeholders for gradient storage
        self.dW = None
        self.db = None
        self.x = None  # cache input for backprop

        # Adam optimizer states
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0  # time step

    def forward(self, x):
        self.x = x
        return np.dot(self.W, x) + self.b

    def backward(self, dy):
        batch_size = self.x.shape[1] if self.x.ndim == 2 else 1

        self.dW = np.dot(dy, self.x.T) / batch_size
        self.db = np.sum(dy, axis=1, keepdims=True) / batch_size
        dx = np.dot(self.W.T, dy)
        return dx

    def update_weights_adam(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1  # Increment time step

        # Update biased first moment estimate
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        # Update biased second raw moment estimate
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        # Compute bias-corrected moment estimates
        mW_hat = self.mW / (1 - beta1 ** self.t)
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)

        # Update parameters
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
        # Reset gradients after update
        self.reset_gradients()

    def reset_gradients(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.dropout_rate == 0.0:
            self.mask = np.ones_like(x)
            return x
        self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(x.dtype)
        return x * self.mask / (1.0 - self.dropout_rate)

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.dropout_rate)

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask