"""Online logistic regression: learns one sample at a time, never needs retraining."""
import numpy as np


class OnlineLogit:
    def __init__(self, n, lr=0.0003, l2=1e-2):
        self.n, self.lr, self.l2 = n, lr, l2
        self.w = np.zeros(n)
        self.b = 0.0
        self.mean = np.zeros(n)
        self.var = np.ones(n)
        self.seen = 0      # feature rows observed (for normalisation)
        self.updates = 0   # labelled samples learned from

    def observe(self, x):
        """Update the running feature normalisation (slowly forgets old regimes)."""
        self.seen += 1
        a = max(1.0 / self.seen, 1.0 / 5000)
        d = x - self.mean
        self.mean += a * d
        self.var += a * (d * (x - self.mean) - self.var)

    def _z(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5)

    def predict(self, x):
        """Probability that the upper barrier is hit before the lower one."""
        return float(1 / (1 + np.exp(-(self._z(x) @ self.w + self.b))))

    def update(self, x, y, weight=1.0):
        z = self._z(x)
        g = (1 / (1 + np.exp(-(z @ self.w + self.b))) - y) * weight
        self.w -= self.lr * (g * z + self.l2 * self.w)
        self.b -= self.lr * g
        self.updates += 1

    def to_dict(self):
        return {"n": self.n, "lr": self.lr, "l2": self.l2, "w": self.w.tolist(), "b": self.b,
                "mean": self.mean.tolist(), "var": self.var.tolist(),
                "seen": self.seen, "updates": self.updates}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["n"], d["lr"], d["l2"])
        m.w, m.mean, m.var = np.array(d["w"]), np.array(d["mean"]), np.array(d["var"])
        m.b, m.seen, m.updates = d["b"], d["seen"], d["updates"]
        return m
