'''
Created on Apr 10, 2015

@author: Yingxiong
'''
'''
Created on Mar 31, 2015

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


# experimental data to be fitted

def plot_scale(m_f, s_f, m_tau, s_tau, w_arr, sig_w):

    def lackoffit(m, sV0, shape, scale, x, data):
        """ model decaying sine wave, subtract data"""

        tau = RV('gamma', shape=shape, scale=scale, loc=0.)
        n_int = 500
        p_arr = np.linspace(0.5 / n_int, 1 - 0.5 / n_int, n_int)
        tau_arr = tau.ppf(p_arr) + 1e-10
        r = 3.5e-3
        E_f = 180e3
        lm = 1000.

        def cdf(e, depsf, r, lm, m, sV0):
            '''weibull_fibers_cdf_mc'''
            s = ((depsf * (m + 1.) * sV0 ** m) /
                 (2. * pi * r ** 2.)) ** (1. / (m + 1.))
            a0 = (e + 1e-15) / depsf
            expfree = (e / s) ** (m + 1.)
            expfixed = a0 / \
                (lm / 2.0) * (e / s) ** (m + 1.) * \
                (1. - (1. - lm / 2.0 / a0) ** (m + 1.))
            mask = a0 < lm / 2.0
            exp = expfree * mask + \
                np.nan_to_num(expfixed * (mask == False))
            return 1. - np.exp(- exp)

        T = 2. * tau_arr / r + 1e-10

        ef0cb = np.sqrt(x[:, np.newaxis] * T[np.newaxis, :] / E_f)
        ef0lin = x[:, np.newaxis] / lm + \
            T[np.newaxis, :] * lm / 4. / E_f
        depsf = T / E_f
        a0 = ef0cb / depsf
        mask = a0 < lm / 2.0
        e = ef0cb * mask + ef0lin * (mask == False)
        Gxi = cdf(e, depsf, r, lm, m, sV0)
        mu_int = e * (1. - Gxi)
        sigma = mu_int * E_f

        residual = np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000 - data

        return np.sum(residual ** 2)

    n = 30
    s_f_arr = np.linspace(0.75 * s_f, 1.25 * s_f, n)
    s_tau_arr = np.linspace(0.75 * s_tau, 1.25 * s_tau, n)

    X, Y = np.meshgrid(s_f_arr, s_tau_arr)

    delta = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            print(j)
            delta[i, j] = lackoffit(
                m_f, X[i, j], m_tau, Y[i, j], w_arr, sig_w)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, delta,  rstride=1, cstride=1, cmap=cm.Greys_r)

    # print 'x'
    # print X
    # print 'y'
    # print Y
    # print 'z'
    # print delta

    plt.figure(figsize=(12, 9))
    im = plt.imshow(delta, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(0.75 * s_f, 1.25 * s_f, 0.75 * s_tau, 1.25 * s_tau), aspect=s_f / s_tau)
    levels = np.arange(0, 5, 0.2)
    CS = plt.contour(delta, levels,
                     origin='lower',
                     linewidths=2,
                     extent=(0.75 * s_f, 1.25 * s_f, 0.75 * s_tau, 1.25 * s_tau))
    plt.clabel(CS, levels[1::2],  # label every second level
               inline=1,
               fmt='%1.1f',
               fontsize=12)
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.title('lack of fit')
    # plt.hot()  # Now change the colormap for the contour lines and colorbar
    plt.flag()

    # We can still add a colorbar for the image, too.
    CBI = plt.colorbar(im, shrink=0.8)

    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.

    # l, b, w, h = plt.gca().get_position().bounds
    # ll, bb, ww, hh = CB.ax.get_position().bounds
    # CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])
    plt.plot(s_f, s_tau, 'ro')
    plt.xlabel('s_f')
    plt.ylabel('s_tau')
    path = 'D:\\fig\\scale'
    plt.savefig(path)

if __name__ == '__main__':

    w_arr = np.linspace(0., 1., 50)
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

    plot_scale(
        13.1, 0.0115, 0.0965, 0.718, w_arr, sig_w)
    #m = 7.7187122824317296
    #sV0 = 0.009961936545108685
#         shape = 0.0505
#         scale = 2.276

    plt.show()
