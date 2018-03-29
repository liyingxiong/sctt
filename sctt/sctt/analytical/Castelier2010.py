'''
Created on 08.02.2018

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def cracking(sig_mu_x, x):
    vf = 0.1
    vm = 1. - vf
    Ef = 67000.  # MPa
    Em = 8250.  # MPa
    Ec = Ef * vf + Em * vm
    alpha = Em * vm / (Ef * vf)
    T = 9.0  # bond intensity 2*tau/r
    sig_cu = 6.

    def sig_m(z, sig_c):  # matrix stress
        sig_m = np.minimum(
            z * T * vf / (1. - vf), Em * sig_c / (vf * Ef + (1. - vf) * Em))
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

            # save the figure
#             plt.figure()
#             plt.plot(x, sig_m(get_z_x(x, XK), sig_c_k))
#             plt.plot(x, sig_mu_x)
#             plt.savefig("D:\\cracking_history\\" + str(len(sig_c_K)) + ".png")
#             plt.close()

        sig_c_K.append(sig_cu)
        eps_c_K.append(np.trapz(eps_f(get_z_x(x, XK), sig_cu), x) / np.amax(x))
        return sig_c_K, eps_c_K

    sig_c_K, eps_c_K = get_cracking_history()

    return sig_c_K, eps_c_K


def curtin(sig_c, L_0, l_slice):

    vf = 0.1
    vm = 1. - vf
    Ef = 67000.  # MPa
    Em = 8250.  # MPa
    Ec = Ef * vf + Em * vm
    tau_r = 4.5
    alpha = Em * vm / (Ef * vf)

    cap_x = 0.88902 * 2.
    sig_0 = 1.0
    sig_0 = sig_0 * Ec / Em
    m = 5.
#     sig_rc = (L_0 / l_slice) ** (1. / m) * sig_0
    sig_rc = sig_0
    print sig_rc

    def delta(sig_c):  # Eq.(9)
        return vm * Em * sig_c / (2. * Ec * vf * tau_r)

    def crack_spacing(sig_c):
        return cap_x / (1. - np.exp(-(sig_c / sig_rc) ** m))

    cs = crack_spacing(sig_c)

    def eps_c(sig_c_arr):
        eps_c = []
        for sig_c in sig_c_arr:
            debonding_l = delta(sig_c)
            cs = crack_spacing(sig_c)
            if cs < 2. * debonding_l:
                eps = sig_c * \
                    (1. / (Ef * vf) - alpha * cs / (4. * debonding_l * Ec))
            else:
                eps = sig_c / Ec * (1. + alpha * debonding_l / cs)
            eps_c.append(eps)
        return np.array(eps_c)

    eps = eps_c(sig_c)
    return cs, eps


sig0 = 1.0
L_c = 1000.  # specimen_length
L_0 = 2.0  # reference_length
# m = 4.9
# m_arr = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 15.])
m_arr = np.array([5.])
n_cracks = []
# n_points_arr = np.array([500, 1000, 5000, 10000, 15000, 20000])
# n_points_arr = np.array([1000])
n_points = 12000
l_slice = L_c / n_points

for m in m_arr:
    x = np.linspace(0, L_c, n_points)
    a = np.random.uniform(size=n_points)
    sig_mu_x = sig0 * (L_0 / l_slice * -np.log(a)) ** (1 / m)
    sig_c_K, eps_c_K = cracking(sig_mu_x, x)
    plt.plot(eps_c_K, sig_c_K, 'k--', label='PMCM')
    n_cracks.append(len(sig_c_K))

print n_cracks
print L_c / (np.array(n_cracks))

# compute the analytical results
sig_c_arr = np.linspace(0, 6, 200)
cs, eps = curtin(sig_c_arr, L_0, l_slice)
plt.plot(eps, sig_c_arr)
plt.figure()
plt.plot(eps, cs)
plt.plot(eps_c_K, L_c / (np.arange(len(sig_c_K)) + 1.), 'k--')
plt.ylim(0, 10)
plt.show()


plt.figure()
m_l = np.loadtxt('D:\\data\\papers\\curtin1991\\m_l.txt').T
m_K = np.loadtxt('D:\\data\\papers\\cuypers2006\\Henstenburg1988.txt').T
m_K[0][-1] = 16.
ax1 = plt.subplot(111)
ax1.plot(m_arr, L_c / (np.array(n_cracks)), 'ro', label='PMCM')
ax1.plot(m_l[0], m_l[1] * 2., 'bs', label='curtin1991')
# ax1.plot(m_K[0], m_K[1], 'k^', label='Henstenburg1988')
ax1.set_ylabel('fragment length / stress recovery length')
# ax1.set_xlabel('number of points')
ax1.set_xlabel('m')
plt.ylim(0.,)
# ax2 = ax1.twinx()
# ax2.plot([500, 1000, 5000, 10000, 15000], 1000. / np.array(n_cracks))
# ax2.set_ylabel('crack spacing')
plt.legend(loc='best')
plt.show()
