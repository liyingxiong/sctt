'''
Created on May 27, 2015

@author: Yingxiong
'''
'''
Created on May 6, 2015

@author: Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.special import gamma
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline

filename = 'D:\\data\\Tensile_test_multiple_cracking\\1.0\\cs.txt'

a = np.loadtxt(filename)
cs = np.reshape(a, (40, 40))

n = 40
s_m_arr = np.linspace(2.4, 3.8, n)
m_m_arr = np.linspace(10, 100, n)

smooth_cs = RectBivariateSpline(s_m_arr, m_m_arr, cs, s=0)

s = np.linspace(2.5, 4.5, 100)
m = np.linspace(5, 35, 100)

cs = smooth_cs(s, m)

filename = 'D:\\data\\Tensile_test_multiple_cracking\\1.0\\lof1.txt'

a = np.loadtxt(filename)
lof = np.reshape(a, (40, 40))

n = 40
s_m_arr = np.linspace(2.4, 3.8, n)
m_m_arr = np.linspace(10, 100, n)

smooth = RectBivariateSpline(s_m_arr, m_m_arr, lof, s=0)

s = np.linspace(2.5, 4.5, 100)
m = np.linspace(5, 35, 100)

lack_of_fit = smooth(s, m)


plt.figure(figsize=(12, 9))
im = plt.imshow(lack_of_fit, interpolation='bilinear', origin='lower',
                cmap=cm.gray_r, extent=(2.5, 4.5, 5, 35), aspect=2 / 30.)
# levels = np.arange(np.amin(lack_of_fit), np.amax(
#     lack_of_fit), (np.amax(lack_of_fit) - np.amin(lack_of_fit)) / 30)
# levels = np.hstack(
#     (np.linspace(9, 19, 11), np.linspace(22 ** 0.5, np.sqrt(np.amax(lack_of_fit)), 10) ** 2))
levels = [5, 6, 7, 8, 9, 11, 13, 15, 17, 20, 24, 29, 35, 41, 47]
print levels
CS = plt.contour(lack_of_fit, levels,
                 origin='lower',
                 cmap=cm.gray,
                 linewidths=1,
                 extent=(2.5, 4.5, 5, 35))
plt.clabel(CS, levels,  # levels[1::1],
           inline=1,
           fmt='%1.1f',
           fontsize=12)

csme = [20.1]
cs2 = plt.contour(cs, csme,
                  linewidths=3,
                  colors='k',
                  extent=(2.5, 4.5, 5, 35), origin='lower')
# plt.clabel(cs2, csme,
#            inline=1,
#            fmt='%1.1f',
#            fontsize=12)
#
#
# plt.title('lack of fit-response curve')
# plt.title('crack spacing')
#
# m_m_arr = np.linspace(10, 100, 100)


def scale(shape):
    lp = 1.
    lc = 1000.
    sig_min = 2.712
    f = (lp / (lp + lc)) ** (1 / shape)
    return sig_min / (f * gamma(1 + 1 / shape))
s_m = scale(m_m_arr)
# plt.plot(s_m[s_m < 3.8], m_m_arr[s_m < 3.8], 'k--')


# CBI = plt.colorbar(im, shrink=0.8)

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


plt.xlabel('s_m')
plt.ylabel('m_m')

plt.show()
