import numpy as np

class optimizer:
    def __init__(self):
        pass

    def __call__(self, params, grads):
        raise NotImplementedError("This optimizer is not implemented")


class SGD(optimizer):
    def __init__(self, lr=0.01):
        self.lr = float(lr)

    def __call__(self, param, grad, param_name=None):
        return param - self.lr * grad.mean(axis=0)
