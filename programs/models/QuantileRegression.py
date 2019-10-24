import numpy as np
from scipy import optimize


class QuantileRegression:
    def __init__(self, y, Phi, quantile):
        self.y = np.array(y).reshape((-1, 1))
        self.Phi = np.array(Phi)
        self.quantile = float(quantile)
        if self.quantile <= 0 or 1 <= self.quantile:
            raise ValueError("quantile q must be in 0 < q < 1")

        self.N, self.D = self.Phi.shape

    def fit(self):
        self._optimizer = optimize.minimize(
            fun=self._calc_loss, x0=np.zeros(self.D)
            )

        self.beta = self._optimizer.x

    def _calc_loss(self, beta):
        pred = self.Phi @ beta
        abs_diff = np.abs(self.y.flatten() - pred)

        loss = 0
        mask = pred <= self.y.flatten()
        loss += self.quantile * np.sum(abs_diff[mask])
        loss += (1 - self.quantile) * np.sum(abs_diff[mask == False])

        return loss

    def predict(self, Phi):
        return Phi @ self.beta
