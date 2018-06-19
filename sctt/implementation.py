'''
Created on 20.12.2017

@author: Yingxiong
'''

import numpy as np
from scipy.optimize import brentq, newton
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


Em = 25e3  # matrix modulus
Ef = 180e3  # fiber modulus
vf = 0.01  # reinforcement ratio
T = 12.  # bond intensity
sig_cu = 18.  # [MPa]
x = np.linspace(0, 5000, 5000)  # specimen discretization
# sig_mu_x = np.linspace(3.0, 3. + 1e-8, 5000)  # matrix strength field
sig_mu_x = np.random.normal(loc=3.0, scale=1e-8, size=5000)
# sig_mu_x = 3. * np.random.weibull(930., size=5000)
vm = 1. - vf

sig_mu = 3.

delta_ack = vm * sig_mu / (T * vf)


Ec = Ef * vf + Em * vm
alpha = Em * vm / (Ef * vf)


def sig_m(z, sig_c):  # matrix stress
    sig_m = np.minimum(
        z * T * vf / (1 - vf), Em * sig_c / (vf * Ef + (1 - vf) * Em))
    return sig_m


def eps_f(z, sig_c):  # reinforcement strain
    sigma_m = sig_m(z, sig_c)
    eps_f = (sig_c - sigma_m * (1. - vf)) / vf / Ef
    return eps_f


def get_z_x(x, XK):  # distance to the closest crack
    z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
    return np.amin(z_grid, axis=1)


def get_sig_c_z(sig_mu, z, sig_c_pre):
    fun = lambda sig_c: sig_mu - sig_m(z, sig_c)
    try:  # search for the local crack load level
        return newton(fun, sig_c_pre)
    # solution not found (shielded zone) return the ultimate composite stress
    except (RuntimeWarning, RuntimeError):
        return sig_cu


def get_sig_c_K(z_x, sig_c_pre):
    get_sig_c_x = np.vectorize(get_sig_c_z)
    sig_c_x = get_sig_c_x(sig_mu_x, z_x, sig_c_pre)  # Eq. (6)
    y_idx = np.argmin(sig_c_x)  # Eq. (7) and Eq.(8)
    return sig_c_x[y_idx], x[y_idx]


def get_cracking_history():
    XK = []  # position of the first crack
    sig_c_K = [0.]
    eps_c_K = [0.]

    idx_0 = np.argmin(sig_mu_x)
    XK.append(x[idx_0])
    sig_c_0 = sig_mu_x[idx_0] * Ec / Em
    sig_c_K.append(sig_c_0)
    eps_c_K.append(sig_mu_x[idx_0] / Em)

    while True:
        z_x = get_z_x(x, XK)
        sig_c_k, y_i = get_sig_c_K(z_x, sig_c_K[-1])
        if sig_c_k == sig_cu:
            break
        XK.append(y_i)
        sig_c_K.append(sig_c_k)
        eps_c_K.append(
            np.trapz(eps_f(get_z_x(x, XK), sig_c_k), x) / np.amax(x))  # Eq. (10)
    sig_c_K.append(sig_cu)
    eps_c_K.append(np.trapz(eps_f(get_z_x(x, XK), sig_cu), x) / np.amax(x))
    return sig_c_K, eps_c_K

sig_c_K, eps_c_K = get_cracking_history()
plt.plot(eps_c_K, sig_c_K)
# plt.plot([0.0, sig_cu / (Ef * vf)], [0.0, sig_cu])

print 5000. / len(sig_c_K)
print 1.337 * delta_ack


def ACK(sig_mu):
    return [0, sig_mu / Em, (1 + 0.666666666 * alpha) * sig_mu / Em], [0, sig_mu / Em * Ec, sig_mu / Em * Ec]


x, y = ACK(sig_mu)
# plt.plot(x, y)


plt.ylim(0, 20)

plt.xlim(0, 0.01)

plt.show()
