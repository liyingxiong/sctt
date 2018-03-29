'''
Created on 28.09.2017

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from stats.misc.random_field.random_field_1D import RandomField
from scipy.interpolate import interp1d

# the m-K relationship from hensenburg1988, m \in (3, 15)
m_K = np.loadtxt('D:\\data\\papers\\cuypers2006\\Henstenburg1988.txt').T
interp_k = interp1d(m_K[0], m_K[1])


sig_rc = 10.5
m = 1000

vf = 0.014
vm = 1. - vf
Ef = 72000.
Em = 18000.
Ec = Ef * vf + Em * vm

alpha = Em * vm / (Ef * vf)

sig_mu = sig_rc * Em / Ec
# delta_ack = vm * sig_mu / (2. * vf * tau_r)
# cap_x = 1.337 * delta_ack

# k = interp_k(m)
# print k
K = 1.337
cap_x = 0.97
tau_r = K * sig_mu * vm / (2 * cap_x * vf)
# print tau_r

delta_ack = vm * sig_mu / (2. * vf * tau_r)


def delta(sig_c):  # Eq.(9)
    return vm * Em * sig_c / (2. * Ec * vf * tau_r)


def crack_spacing(sig_c):
    return cap_x / (1. - np.exp(-(sig_c / sig_rc) ** m))


# def eps_c(sig_c):
#     cs = x(sig_c)
#     debonding_l = delta(sig_c)
#     case_idx = cs < 2. * debonding_l
#     eps_c1 = sig_c / Ec * (1 + alpha * debonding_l / cs)
#     eps_c2 = sig_c * (1. / (Ef * vf) - alpha * cs / (4. * debonding_l * Ec))
#     eps = eps_c1 * (1. - case_idx) + eps_c2 * case_idx
#     return eps

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


def ACK(sig_mu):

    return [0, sig_mu / Em, (1 + 0.666 * alpha) * sig_mu / Em], [0, sig_mu / Em * Ec, sig_mu / Em * Ec]

sig_c = np.linspace(0, 20, 1000)

# plt.plot(sig_c, cs)
# theory_cs = np.loadtxt('D:\\data\\papers\\cuypers2006\\theory_cs.txt').T
# plt.plot(theory_cs[0], theory_cs[1])
# plt.ylim(0, 5)
# plt.figure()

eps = eps_c(sig_c)
plt.plot(eps, sig_c, label='analytical')
theory = np.loadtxt('D:\\data\\papers\\cuypers2006\\theory.txt').T
# plt.plot(theory[0] / 100., theory[1], label='analytical')
expri = np.loadtxt('D:\\data\\papers\\cuypers2006\\1.txt').T
plt.plot(expri[0] / 100., expri[1], label='test')
# plt.legend()
# plt.show()


# x, y = ACK(sig_rc / Ec * Em)
#
# plt.plot(x, y, label='ACK')
# plt.legend()
# plt.show()

T = 2 * tau_r  # bond intensity 2*tau/r
sig_cu = 20.  # [MPa]
x = np.linspace(0, 300, 1500)  # specimen discretization
# sig_mu_x = np.linspace(3.0, 3. + 1e-8, 5000)  # matrix strength field
# sig_mu_x = np.random.normal(loc=sig_rc, scale=m, size=2000)
sig_mu_x = sig_rc * Em / Ec * np.random.weibull(m, size=1500)
# plt.figure()
# plt.plot(x, sig_mu_x, label='1')
sig_mu_x = sig_rc * np.random.weibull(m, size=2000)
random_field = RandomField(seed=False,
                           lacor=0.1,
                           length=300,
                           nx=1500,
                           nsim=1,
                           loc=.0,
                           shape=m,
                           scale=sig_rc * Em / Ec,
                           distr_type='Weibull')
sig_mu_x = random_field.random_field
# plt.plot(x, sig_mu_x, label='2')
# plt.legend()
# plt.show()


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
    # solution not found (shielded zone) return the ultimate composite stress
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
    eps_c_K.append(np.trapz(eps_f(get_z_x(x, XK), sig_cu), x) / np.amax(x))
    return sig_c_K, eps_c_K

sig_c_K, eps_c_K = get_cracking_history()
plt.plot(eps_c_K, sig_c_K, label='pmcm')
plt.legend()


print np.amax(x)
print len(sig_c_K)
cs = np.amax(x) / len(sig_c_K)
print cs / delta_ack
print cs

plt.figure()
for i in np.arange(7):
    cs_e = np.loadtxt(
        'D:\\data\\papers\\cuypers2006\\' + str(i + 2) + '.txt').T
    plt.plot(cs_e[0], cs_e[1], 'gray')
cs = crack_spacing(sig_c)  # theoretical cs
plt.plot(sig_c, cs)
plt.plot(sig_c_K, np.amax(x) / (np.arange(len(sig_c_K)) + 1.), 'k--')
plt.ylim(0, 5)
plt.xlim(5, 20)

# print len(sig_c_K)

plt.show()
