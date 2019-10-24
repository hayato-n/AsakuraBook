import numpy as np


class AffineLayer:
    def __init__(self, n_input, n_output, init="Xavier"):
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.init = str(init)

        self.w = self._init_weight(n_input, n_output, method=self.init).T
        self.b = self._init_weight(n_output, 1, method=self.init).T

    def _init_weight(self, n_input, n_output, method="Xavier"):
        if method == "Xavier":
            devide = np.sqrt(n_input)
        elif method == "He":
            devide = np.sqrt(n_input/2)
        else:
            raise NotImplementedError("Undefined weight initialization method")

        return np.random.randn(n_input, n_output) / devide

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x
        self.output = self.input @ self.w.T + self.b
        return self.output

    def backward(self, backproped):
        self.grad = {
            "w": self.input[:, np.newaxis, :] * backproped[..., np.newaxis],
            "b": 1 * backproped
        }
        self.gradh = np.sum(
            self.w[np.newaxis, :, 0] * backproped[:, np.newaxis, :],
            axis=1
        )

        return self.gradh


class LinearLayer:
    def __init__(self, affine, activation):
        self.affine = affine
        self.activation = activation

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x
        self.output = self.activation(self.affine(x))
        return self.output

    def backward(self, backproped):
        backproped = self.activation.backward(backproped)
        backproped = self.affine.backward(backproped)
        return backproped
