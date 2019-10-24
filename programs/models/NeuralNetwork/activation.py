import numpy as np


# activation functions
class activation:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError("forward process is not available")

    def backward(self, backproped):
        raise NotImplementedError("backward process is not available")


class Sigmoid(activation):
    def __init__(self, threshold=1e-3):
        self._threshold = float(threshold)
        super().__init__()

    def forward(self, x):
        self.input = x
        mask = x > self._threshold
        self.output = np.zeros_like(x, dtype=float)
        self.output[mask] = 1 / (1 + np.exp(-x[mask]))
        return self.output

    def backward(self, backproped):
        self.grad = self.output * (1 - self.output) * backproped
        return self.grad


class ReLU(activation):
    def forward(self, x):
        self.input = x
        mask = x > 0
        self.output = np.zeros_like(x, dtype=float)
        self.output[mask] = x[mask]
        return self.output

    def backward(self, backproped):
        self.grad = np.zeros_like(self.input, dtype=float)
        self.grad[self.input > 0] = 1
        self.grad *= backproped
        return self.grad

