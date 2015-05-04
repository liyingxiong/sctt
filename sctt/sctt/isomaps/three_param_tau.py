'''
Created on Apr 22, 2015

@author: Yingxiong
'''
import numpy as np
import os
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from spirrid.rv import RV
from math import pi
import time as t
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_m_s(s_tau, m_tau, l_tau, w_arr, sig_w):

    def lackoffit(shape, scale, loc, x, data):

        sV0 = 0.0076
        m = 6.7
        tau = RV('gamma', shape=shape, scale=scale, loc=loc)
        n_int = 500
        p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
        tau_arr = tau.ppf(p_arr) + 1e-10
        r = 3.5e-3
        E_f = 180e3
        T = 2. * tau_arr / r
        # scale parameter with respect to a reference volume
        s = ((T * (m + 1.) * sV0 ** m) /
             (2. * E_f * pi * r ** 2)) ** (1. / (m + 1.))
        ef0 = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1.))
        mu_int = ef0 * (1 - Gxi)
        sigma = mu_int * E_f

        residual = np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000 - data

        return np.sum(residual ** 2)

    n = 30
    m_tau_arr = np.linspace(0.75 * m_tau, 1.25 * m_tau, n)
    s_tau_arr = np.linspace(0.75 * s_tau, 1.25 * s_tau, n)

    X, Y = np.meshgrid(m_tau_arr, s_tau_arr)

    delta = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            print j
            delta[i, j] = lackoffit(
                X[i, j], Y[i, j], l_tau, w_arr, sig_w)

    plt.figure(figsize=(12, 9))
    im = plt.imshow(delta, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(0.75 * m_tau, 1.25 * m_tau, 0.75 * s_tau, 1.25 * s_tau), aspect=m_tau / s_tau)
    levels = np.arange(0, 10, 0.2)
    CS = plt.contour(delta, levels,
                     origin='lower',
                     linewidths=1,
                     extent=(0.75 * m_tau, 1.25 * m_tau, 0.75 * s_tau, 1.25 * s_tau))
    plt.clabel(CS, levels[1::2],  # label every second level
               inline=1,
               fmt='%1.1f',
               fontsize=12)
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.title('lack of fit')
    plt.flag()

    CBI = plt.colorbar(im, shrink=0.8)

    plt.plot(m_tau, s_tau, 'ro')
    plt.xlabel('m_tau')
    plt.ylabel('s_tau')
    path = 'D:\\fig\\m_s'
    plt.savefig(path)


def plot_m_l(s_tau, m_tau, l_tau, w_arr, sig_w):

    def lackoffit(shape, scale, loc, x, data):

        sV0 = 0.0076
        m = 6.7
        tau = RV('gamma', shape=shape, scale=scale, loc=loc)
        n_int = 500
        p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
        tau_arr = tau.ppf(p_arr) + 1e-10
        r = 3.5e-3
        E_f = 180e3
        T = 2. * tau_arr / r
        # scale parameter with respect to a reference volume
        s = ((T * (m + 1.) * sV0 ** m) /
             (2. * E_f * pi * r ** 2)) ** (1. / (m + 1.))
        ef0 = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1.))
        mu_int = ef0 * (1 - Gxi)
        sigma = mu_int * E_f

        residual = np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000 - data

        return np.sum(residual ** 2)

    n = 30
    m_tau_arr = np.linspace(0.75 * m_tau, 1.25 * m_tau, n)
    l_tau_arr = np.linspace(0.75 * l_tau, 1.25 * l_tau, n)

    X, Y = np.meshgrid(m_tau_arr, l_tau_arr)

    delta = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            print j
            delta[i, j] = lackoffit(
                X[i, j], s_tau, Y[i, j], w_arr, sig_w)

    plt.figure(figsize=(12, 9))
    im = plt.imshow(delta, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(0.75 * m_tau, 1.25 * m_tau, 0.75 * l_tau, 1.25 * l_tau), aspect=m_tau / l_tau)
    levels = np.arange(0, 10, 0.2)
    CS = plt.contour(delta, levels,
                     origin='lower',
                     linewidths=1,
                     extent=(0.75 * m_tau, 1.25 * m_tau, 0.75 * l_tau, 1.25 * l_tau))
    plt.clabel(CS, levels[1::2],  # label every second level
               inline=1,
               fmt='%1.1f',
               fontsize=12)
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.title('lack of fit')
    plt.flag()

    CBI = plt.colorbar(im, shrink=0.8)

    plt.plot(m_tau, l_tau, 'ro')
    plt.plot((0.75 * m_tau, 1.25 * m_tau), (0.00126, 0.00126), '--')
    plt.xlabel('m_tau')
    plt.ylabel('l_tau')
    path = 'D:\\fig\\m_l'
    plt.savefig(path)


def plot_s_l(s_tau, m_tau, l_tau, w_arr, sig_w):

    def lackoffit(shape, scale, loc, x, data):

        sV0 = 0.0076
        m = 6.7
        tau = RV('gamma', shape=shape, scale=scale, loc=loc)
        n_int = 500
        p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
        tau_arr = tau.ppf(p_arr) + 1e-10
        r = 3.5e-3
        E_f = 180e3
        T = 2. * tau_arr / r
        # scale parameter with respect to a reference volume
        s = ((T * (m + 1.) * sV0 ** m) /
             (2. * E_f * pi * r ** 2)) ** (1. / (m + 1.))
        ef0 = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (m + 1.))
        mu_int = ef0 * (1 - Gxi)
        sigma = mu_int * E_f

        residual = np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000 - data

        return np.sum(residual ** 2)

    n = 10
    s_tau_arr = np.linspace(0.75 * s_tau, 1.25 * s_tau, n)
    l_tau_arr = np.linspace(0.75 * l_tau, 1.25 * l_tau, n)

    X, Y = np.meshgrid(s_tau_arr, l_tau_arr)

    delta = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            print j
            delta[i, j] = lackoffit(
                m_tau, X[i, j], Y[i, j], w_arr, sig_w)

    plt.figure(figsize=(12, 9))
    im = plt.imshow(delta, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(0.75 * s_tau, 1.25 * s_tau, 0.75 * l_tau, 1.25 * l_tau), aspect=s_tau / l_tau)
    levels = np.arange(0, 10, 0.2)
    CS = plt.contour(delta, levels,
                     origin='lower',
                     linewidths=1,
                     extent=(0.75 * s_tau, 1.25 * s_tau, 0.75 * l_tau, 1.25 * l_tau))
    plt.clabel(CS, levels[1::2],  # label every second level
               inline=1,
               fmt='%1.1f',
               fontsize=12)
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.title('lack of fit')
    plt.flag()

    CBI = plt.colorbar(im, shrink=0.8)

    plt.plot(s_tau, l_tau, 'ro')
    plt.plot((0.75 * s_tau, 1.25 * s_tau), (0.00126, 0.00126), '--')
    plt.xlabel('s_tau')
    plt.ylabel('l_tau')
    path = 'D:\\fig\\s_l'
    plt.savefig(path)


if __name__ == '__main__':

    w_arr = np.linspace(0., 1.0, 100)
    sig_w = np.zeros_like(w_arr)
    home_dir = 'D:\\Eclipse\\'
    for i in np.array([1, 2, 3, 4, 5]):
        path = [home_dir, 'git',  # the path of the data file
                'rostar',
                'scratch',
                'diss_figs',
                'CB' + str(i) + '.txt']
        filepath = os.path.join(*path)
    #     exp_data = np.zeros_like(w_arr)
        file1 = open(filepath, 'r')
        cb = np.loadtxt(file1, delimiter=';')
        test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
        test_ydata = cb[:, 1]
        interp = interp1d(
            test_xdata, test_ydata, bounds_error=False, fill_value=0.)
        sig_w += 0.2 * interp(w_arr)

    plot_m_s(1.440, 0.0539, 0.001260, w_arr, sig_w)

    plt.show()
