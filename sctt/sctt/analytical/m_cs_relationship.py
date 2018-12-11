'''
Created on 06.02.2018

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from stats.misc.random_field.random_field_1D import RandomField
sig_rc = 10.5
m = 1000

vf = 0.104
vm = 1. - vf
Ef = 72000.
Em = 18000.
Ec = Ef * vf + Em * vm

alpha = Em * vm / (Ef * vf)

tau_r = 63.2

sig_mu = sig_rc * Em / Ec
T = 2 * tau_r  # bond intensity 2*tau/r
sig_cu = 20.  # [MPa]
x = np.linspace(0, 75, 1500)  # specimen discretization

delta_ack = vm * sig_mu / (2. * vf * tau_r)

print(delta_ack)

cs = []
m_arr = np.array([2., 4., 6., 10., 14., 3000.])
# m_arr = np.array([3000.])

for m in m_arr:
    random_field = RandomField(seed=False,
                               lacor=1000.,
                               length=75,
                               nx=1500,
                               nsim=1,
                               loc=.0,
                               shape=m,
                               scale=sig_rc * Em / Ec,
                               distr_type='Weibull')
    sig_mu_x = random_field.random_field

#     plt(x., )

    def cb(z, sig_c):  # Eq.(3) and Eq. (9)
        sig_m = np.minimum(
            z * T * vf / (1. - vf), Em * sig_c / (vf * Ef + (1. - vf) * Em))  # matrix stress
        eps_f = (sig_c - sig_m * (1. - vf)) / vf / Ef  # reinforcement strain
        return sig_m, eps_f

    def get_z_x(x, XK):  # Eq.(5)
        z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
        return np.amin(z_grid, axis=1)

    def get_lambda_z(sig_mu, z):
        fun = lambda sig_c: sig_mu - cb(z, sig_c)[0]
        try:  # search for the local crack load level
            return brentq(fun, 0, sig_cu)
        # solution not found (shielded zone) return the ultimate composite
        # stress
        except:
            return sig_cu

    def get_sig_c_K(z_x):
        get_lambda_x = np.vectorize(get_lambda_z)
        lambda_x = get_lambda_x(sig_mu_x, z_x)  # Eq. (6)
        y_idx = np.argmin(lambda_x)  # Eq. (7) and Eq.(8)
        return lambda_x[y_idx], x[y_idx]

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
            sig_c_k, y_i = get_sig_c_K(z_x)
            if sig_c_k == sig_cu:
                break
            XK.append(y_i)
            sig_c_K.append(sig_c_k)
            eps_c_K.append(
                np.trapz(cb(get_z_x(x, XK), sig_c_k)[1], x) / np.amax(x))  # Eq. (10)
        sig_c_K.append(sig_cu)
        eps_c_K.append(
            np.trapz(cb(get_z_x(x, XK), sig_cu)[1], x) / np.amax(x))
        return sig_c_K, eps_c_K

    sig_c_K, eps_c_K = get_cracking_history()
    plt.plot(eps_c_K, sig_c_K, label='m=' + str(m))

    cs.append(np.amax(x) / len(sig_c_K))

plt.legend()

fig, ax = plt.subplots()
print(cs)
print((np.array(cs) / delta_ack))
m_arr[-1] = 20
ax.plot(m_arr, np.array(cs) / delta_ack)
ax.set_xticks([i for i in m_arr])
m_arr[-1] = 3000
ax.set_xticklabels(m_arr)
plt.show()
