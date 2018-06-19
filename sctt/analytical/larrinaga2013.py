'''
Created on 15.01.2018

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from stats.misc.random_field.random_field_1D import RandomField

tb2_lower = np.loadtxt(
    'D:\\data\\papers\\larrinaga2013\\tb2_lower.txt', delimiter=',').T
tb2_upper = np.loadtxt(
    'D:\\data\\papers\\larrinaga2013\\tb2_upper.txt', delimiter=',').T
tb2_x = np.linspace(0, 0.0241, 200)
tb2_y_lower = np.interp(tb2_x, tb2_lower[0] / 100., tb2_lower[1])
tb2_y_upper = np.interp(tb2_x, tb2_upper[0] / 100., tb2_upper[1])
# plt.fill_between(
#     tb2_x, tb2_y_lower, tb2_y_upper, color='black', alpha=0.3, label='TB2')


tb3_lower = np.loadtxt(
    'D:\\data\\papers\\larrinaga2013\\tb3_lower.txt', delimiter=',').T
tb3_upper = np.loadtxt(
    'D:\\data\\papers\\larrinaga2013\\tb3_upper.txt', delimiter=',').T
tb3_x = np.linspace(0, 0.0261, 200)
tb3_y_lower = np.interp(tb3_x, tb3_lower[0] / 100., tb3_lower[1])
tb3_y_upper = np.interp(tb3_x, tb3_upper[0] / 100., tb3_upper[1])
# plt.fill_between(
#     tb3_x, tb3_y_lower, tb3_y_upper, color='black', alpha=0.5, label='TB3')

tb4_lower = np.loadtxt(
    'D:\\data\\papers\\larrinaga2013\\tb4_lower.txt', delimiter=',').T
tb4_upper = np.loadtxt(
    'D:\\data\\papers\\larrinaga2013\\tb4_upper.txt', delimiter=',').T
tb4_x = np.linspace(0, 0.0241, 200)
tb4_y_lower = np.interp(tb4_x, tb4_lower[0] / 100., tb4_lower[1])
tb4_y_upper = np.interp(tb4_x, tb4_upper[0] / 100., tb4_upper[1])
# plt.fill_between(
#     tb4_x, tb4_y_lower, tb4_y_upper, color='black', alpha=0.7, label='TB4')


n_tb2 = 5. + 6. + 4. + 7. + 6. + 5. + 4.
n_tb3 = 8. + 8. + 8. + 5. + 9. + 8. + 8.
n_tb4 = 10. + 10. + 9. + 9. + 10. + 8. + 9.

cs_tb2 = 210. * 7. / n_tb2
cs_tb3 = 210. * 7. / n_tb3
cs_tb4 = 210. * 7. / n_tb4

print cs_tb2, cs_tb3, cs_tb4

# PMCM-ACK
for vf in [0.007, 0.0105, 0.014]:
    # vf = 0.014
    vm = 1. - vf
    Ef = 67000.  # MPa
    Em = 8250.  # MPa
    Ec = Ef * vf + Em * vm
    alpha = Em * vm / (Ef * vf)
# T = 11.842  # bond intensity 2*tau/r
    sig_cu = 21.2 * vf / 0.014  # [MPa]
    x = np.linspace(0, 5000, 5000)  # specimen discretization
    # matrix strength field
    sig_mu = 2.48
    sig_mu_x = np.random.normal(loc=sig_mu, scale=1e-8, size=5000)

    def ACK(sig_mu):
        # ACK model
        return [0, sig_mu / Em, (1 + 0.666 * alpha) * sig_mu / Em, (1 + 0.666 * alpha) * sig_mu / Em + (sig_cu - sig_mu * Ec / Em) / (Ef * vf)], [0, sig_mu / Em * Ec, sig_mu / Em * Ec, sig_cu]
    ack_eps, ack_sig = ACK(sig_mu)
#     plt.plot(ack_eps, ack_sig, label='ACK')
    t = np.linspace(5, 25, 100)
    delta = vm * sig_mu / (t * vf)
    cs = 1.337 * delta
    plt.plot(t, cs)

    T_arr = [5., 10., 15., 20., 25.]
    cs_arr = []

    for T in T_arr:

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
                return brentq(fun, 0, sig_cu)
            # solution not found (shielded zone) return the ultimate composite
            # stress
            except:
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
            eps_c_K.append(
                np.trapz(eps_f(get_z_x(x, XK), sig_cu), x) / np.amax(x))
            return sig_c_K, eps_c_K

        sig_c_K, eps_c_K = get_cracking_history()
        plt.plot(eps_c_K, sig_c_K, label='PMCM')
        # plt.plot([0.0, sig_cu / (Ef * vf)], [0.0, sig_cu])

        cs = 5000. / (len(sig_c_K) - 2.)
        cs_arr.append(cs)

    plt.plot(T_arr, cs_arr, 'ro')

# plt.legend(loc='upper left')
plt.show()
