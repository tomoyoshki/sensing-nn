import numpy as np


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs = []
    for i in range(X.shape[1]):
        cs.append(CubicSpline(xx[:, i], yy[:, i])(x_range))
    return np.array(cs).transpose()
