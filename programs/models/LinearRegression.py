import numpy as np
# from scipy import stats, optimize


def PolyBaseFunction(x, dims):
    return np.array([x**d for d in range(dims+1)]).T


class LinearRegression:
    def __init__(self, y, Phi):
        self.y = np.array(y).reshape((-1, 1))
        self.Phi = np.array(Phi)

        self.N, self.D = self.Phi.shape

    def fit_OLS(self):
        self.beta_OLS \
            = np.linalg.pinv(self.Phi.T @ self.Phi) @ self.Phi.T @ self.y
        diff = self.y - self.predict_OLS(self.Phi)
        self._loss_OLS = float(diff.T @ diff)

    def predict_OLS(self, Phi):
        return Phi @ self.beta_OLS

    def fit_ML(self):
        self.fit_OLS()
        self.beta_ML = self.beta_OLS
        self.sigma2_ML = self._loss_OLS / self.N
        self._loglik_ML = -0.5*(self._loss_OLS / self.sigma2_ML
                                + self.N * np.log(2*np.pi)
                                + self.N * np.log(self.sigma2_ML))

    def predict_ML(self, Phi):
        return self.predict_OLS(Phi)

    def fit_Ridge(self, alpha):
        self.alpha_Ridge = float(alpha)
        inversed = np.linalg.pinv(self.Phi.T @ self.Phi
                                  + self.alpha_Ridge*np.eye(self.D))
        self.beta_Ridge \
            = inversed @ self.Phi.T @ self.y
        diff = self.y - self.predict_Ridge(self.Phi)
        self._loss_Ridge = float(diff.T @ diff) \
            + self.alpha_Ridge * self._penalty_Ridge(self.beta_Ridge)

    def predict_Ridge(self, Phi):
        return Phi @ self.beta_Ridge

    def _penalty_Ridge(self, beta):
        return float(beta.T @ beta)

    def fit_Bayes(self, sigma2, prior_cov):
        self.sigma2_Bayes = float(sigma2)
        self.prior_cov_Bayes = np.array(prior_cov).reshape((self.D, self.D))
        self._prior_prec_Bayes = np.linalg.pinv(self.prior_cov_Bayes)

        self._beta_prec_Bayes \
            = self._prior_prec_Bayes \
            + self.Phi.T @ self.Phi / self.sigma2_Bayes
        self.beta_cov_Bayes = np.linalg.pinv(self._beta_prec_Bayes)
        self.beta_mean_Bayes = \
            self.beta_cov_Bayes @ self.Phi.T @ self.y / self.sigma2_Bayes

    def predict_Bayes(self, Phi):
        N = len(Phi)

        mu = (Phi @ self.beta_mean_Bayes).flatten()
        sigma2 = np.empty(N)
        for n in range(N):
            sigma2[n] = 1 / self.sigma2_Bayes \
                + Phi[n].T @ self.beta_cov_Bayes @ Phi[n]

        return mu, sigma2
