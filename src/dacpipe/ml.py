"""Gaussian process on atomic features, validated by leaving one metal out at a time."""
import warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (ConstantKernel as C, DotProduct, Matern,
                                              WhiteKernel)

warnings.filterwarnings('ignore')


def make_kernel(n_feat, linear=True):
    # ARD Matern 5/2 for the non-linear part, optional linear term for trends along a series
    k = C(1.0, (1e-2, 1e2)) * Matern(np.full(n_feat, 4.0), (0.3, 200), nu=2.5) + WhiteKernel(0.05, (1e-4, 1))
    if linear:
        k = k + C(0.1, (1e-3, 10)) * DotProduct(1.0, (1e-2, 1e2))
    return k


class Model:
    def __init__(self, linear=True, restarts=0, seed=0):
        self.linear, self.restarts, self.seed = linear, restarts, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.gp = GPR(make_kernel(X.shape[1], self.linear), normalize_y=True,
                      n_restarts_optimizer=self.restarts, random_state=self.seed)
        self.gp.fit((X - self.mu) / self.sd, y)
        return self

    def predict(self, X):
        return self.gp.predict((np.asarray(X, float) - self.mu) / self.sd, return_std=True)

    def length_scales(self):
        """1/l per feature, a rough relevance ranking (only meaningful with the Matern part)"""
        k = self.gp.kernel_
        for part in k.get_params().values():
            if isinstance(part, Matern):
                return 1.0 / np.atleast_1d(part.length_scale)
        return None


def _one_fold(X, y, m1, m2, metal, linear):
    test = (m1 == metal) | (m2 == metal)
    mdl = Model(linear).fit(X[~test], y[~test])
    mu, sg = mdl.predict(X[test])
    return np.flatnonzero(test), mu, sg


def lomo(X, y, m1, m2, linear=True, n_jobs=2):
    """Leave-one-metal-out. Each pair has two metals so it is predicted twice and averaged."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m1, m2 = np.asarray(m1), np.asarray(m2)
    metals = sorted(set(m1) | set(m2))
    folds = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(X, y, m1, m2, m, linear) for m in metals)
    mu, var, cnt = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for idx, a, b in folds:
        mu[idx] += a
        var[idx] += b ** 2
        cnt[idx] += 1
    mu, sg = mu / cnt, np.sqrt(var / cnt)
    res = mu - y
    z = res / sg
    return dict(mu=mu, sg=sg, rmse=float(np.sqrt(np.mean(res ** 2))),
                r2=float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)),
                cov68=float(np.mean(np.abs(z) < 1)), cov95=float(np.mean(np.abs(z) < 1.96)),
                z_std=float(z.std()))


def calibrate(sg, z_std):
    """Rescale predicted standard deviations by the LOMO z-score spread."""
    return np.asarray(sg) * z_std


def exceed_probability(mu, sg, limit):
    """P(value <= limit) for a Gaussian prediction; used to rank unlabelled pairs."""
    from scipy.stats import norm
    return norm.cdf((limit - np.asarray(mu)) / np.asarray(sg))


def active_learning(X, y, n_start=15, n_add=5, rounds=10, strategy='uncertainty', seed=0):
    """Pick pairs for the next batch of DFT. `y` plays the role of the calculation here, so
    this is for benchmarking the strategy on a library that is already labelled."""
    rng = np.random.default_rng(seed)
    n = len(y)
    have = list(rng.choice(n, n_start, replace=False))
    curve = []
    for _ in range(rounds):
        mdl = Model(linear=True).fit(X[have], y[have])
        mu, sg = mdl.predict(X)
        rest = np.setdiff1d(np.arange(n), have)
        curve.append((len(have), float(np.sqrt(np.mean((mu[rest] - y[rest]) ** 2)))))
        if strategy == 'uncertainty':
            pick = rest[np.argsort(sg[rest])[-n_add:]]
        else:
            pick = rng.choice(rest, n_add, replace=False)
        have += list(pick)
    return curve
