import numpy as np


class MeanSquareError:
    def __init__(self):
        pass

    def __call__(self, pred, obs):
        return self.forward(pred, obs)

    def forward(self, pred, obs):
        self.pred = pred.flatten()
        self.obs = obs.flatten()
        self.loss = np.mean(np.square(self.pred - self.obs))
        return self.loss

    def backward(self):
        self.grad = -2 * (self.obs - self.pred)
        return self.grad.reshape((-1, 1))
