import numpy as np
from scipy import signal


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype('float32')


def gaussian(mu, sigma, sample=True):
    if sample:
        return np.random.normal(mu, sigma).astype('float32')
    else:
        return mu.astype('float32')


def elu(x):
    return np.where(x > 0, x, np.expm1(x))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softplus(x):
    return np.log(1. + np.exp(x))