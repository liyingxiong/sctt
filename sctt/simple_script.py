import numpy as np
from scipy.optimize import brentq
from matplotlib import pyplot as plt

Em = 10e3  # matrix modulus
Ef = 180e3  # fiber modulus
vf = 0.01  # reinforcement ratio
T = 12.  # bond intensity
sig_cu = 18.  # [MPa]
L = 1000.
x = np.linspace(0, L, 1000)  # specimen discretization
sig_mu_x = np.linspace(3.0, 3.00000001, 1000)  # matrix strength field


def cb(z, sig_c):  # Eq.(3) and Eq. (9)
    sig_m = np.minimum(
        z * T * vf / (1 - vf), Em * sig_c / (vf * Ef + (1 - vf) * Em))  # matrix stress
    esp_f = (sig_c - sig_m * (1 - vf)) / vf / Ef  # reinforcement strain
    return sig_m, esp_f

# z = np.linspace(-25, 25, 2001)
# sig_m, eps_f = cb(np.abs(z), 3.0)
# plt.plot(z, sig_m / Em, 'b')
# plt.plot(z, eps_f, 'b', label='low bond')
#
# T = 24.
# sig_m, eps_f = cb(np.abs(z), 3.0)
# plt.plot(z, sig_m / Em, 'k', label='high bond')
# plt.plot(z, eps_f, 'k')
#
# plt.legend()
# plt.show()


def get_z_x(x, XK):  # Eq.(5)
    z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
    return np.amin(z_grid, axis=1)


def get_lambda_z(sig_mu, z):
    fun = lambda sig_c: sig_mu - cb(z, sig_c)[0]
    try:  # search for the local crack load level
        return brentq(fun, 0, sig_cu)
    # solution not found (shielded zone) return the ultimate composite stress
    except:
        return sig_cu


def get_sig_c_K(z_x):
    get_lambda_x = np.vectorize(get_lambda_z)
    lambda_x = get_lambda_x(sig_mu_x, z_x)  # Eq. (6)
    y_idx = np.argmin(lambda_x)  # Eq. (7) and Eq.(8)
    return lambda_x[y_idx], x[y_idx]


def get_cracking_history():
    # XK = [0.]  # position of the first crack
    #     sig_c_K = [0., 3.0]
    #     eps_c_K = [0., 3.0 / (vf * Ef + (1 - vf) * Em)]

    XK = []
    sig_c_K = [0.]
    eps_c_K = [0.]

    cs = 20.

    # introduce the predefined cracks
    cracks = cs * np.arange(L / cs + 1)
    d = np.abs(x[:, None] - cracks[None, :])
    min_idx = np.argmin(d, axis=0)
    crack_postion = x[min_idx]

    for i, crack in enumerate(crack_postion):
        XK.append(crack)
        # add a small number to the min matrix strength to avoid numerical
        # problems
        sig_c_K.append(3.0 + i * 1e-8)
        eps_c_K.append(
            np.trapz(cb(get_z_x(x, XK), 3.0 + i * 1e-8)[1], x) / 1000.)  # Eq. (10)

    for sig_c in np.linspace(3, sig_cu, 1000):
        sig_c_K.append(sig_c)
        eps_c_K.append(np.trapz(cb(get_z_x(x, XK), sig_c)[1], x) / 1000.)
    return sig_c_K, eps_c_K

sig_c_K, eps_c_K = get_cracking_history()
plt.plot(eps_c_K, sig_c_K, label='low_bond')


T = 24.  # bond intensity

sig_c_K, eps_c_K = get_cracking_history()
plt.plot(eps_c_K, sig_c_K, label='high_bond')

plt.legend()
# plt.plot([0.0, sig_cu/(Ef*vf)], [0.0, sig_cu])
# plt.axis('off')
plt.show()
