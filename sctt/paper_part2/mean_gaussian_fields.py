'''
Created on 30.11.2016

@author: Yingxiong
'''
import numpy as np


def get_mu_k(mean, std, k):
    w = np.log(k)
    return mean + std * (-0.007 * w ** 3. + 0.1025 * w ** 2. - 0.8684 * w)


def get_mu_k1(mean, std, k):
    w = np.log(k)
    a = std / np.sqrt(2 * w)
    b = mean + std * \
        ((np.log(w) + np.log(4. * np.pi)) / np.sqrt(8. * w) - np.sqrt(2. * w))
    return b - 0.577 * a

std = np.linspace(0.03, 0.45, 100)

from matplotlib import pyplot as plt

plt.plot(std, get_mu_k(3.0, std, 500))
plt.plot(std, get_mu_k1(3.0, std, 500))
plt.show()
