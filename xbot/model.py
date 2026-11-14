"""Gradient-boosted trees that predict the net result (in R) of a setup.

Trees can learn conditional relations a linear model cannot ("RSI 30 means something
different at ADX 40 than at ADX 15"). The model is refitted periodically on all outcomes
known so far, which is how the bot keeps learning.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class SetupModel:
    def __init__(self):
        self.reg = None
        self.n_train = 0

    def fit(self, X, r):
        X = np.array(X, np.float32)
        X[:, np.isnan(X).all(axis=0)] = 0   # a feature without any data cannot be binned
        self.reg = HistGradientBoostingRegressor(
            max_depth=3, max_iter=150, learning_rate=0.03, min_samples_leaf=100,
            l2_regularization=1.0, random_state=0).fit(X, r)
        self.n_train = len(r)

    def predict(self, x):
        """Expected net R of one candidate, or None while there is no model yet."""
        if self.reg is None:
            return None
        return float(self.reg.predict(np.asarray(x, np.float32).reshape(1, -1))[0])
