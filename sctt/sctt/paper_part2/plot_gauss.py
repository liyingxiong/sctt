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

filename = 'D:\\data\\Tensile_test_multiple_cracking\\gauss\\cs.txt'

a = np.loadtxt(filename)
cs = np.reshape(a, (12, 12))

n = 12
s_m_arr = np.linspace(2.3, 3.8, n)
m_m_arr = np.linspace(0.03, 0.45, n)

smooth_cs = RectBivariateSpline(s_m_arr, m_m_arr, cs, s=200)

s = np.linspace(2.3, 3.8, 100)
m = np.linspace(0.03, 0.45, 100)

cs = smooth_cs(s, m)

filename = 'D:\\data\\Tensile_test_multiple_cracking\\gauss\\lof_avg.txt'

a = np.loadtxt(filename)
lof = np.reshape(a, (12, 12))


n = 12
s_m_arr = np.linspace(2.3, 3.8, n)
m_m_arr = np.linspace(0.03, 0.45, n)

smooth = RectBivariateSpline(s_m_arr, m_m_arr, lof, s=200)


s = np.linspace(2.3, 3.8, 100)
m = np.linspace(0.03, 0.45, 100)

lack_of_fit = smooth(s, m)


plt.figure(figsize=(12, 9))
im = plt.imshow(lack_of_fit, interpolation='bilinear', origin='lower',
                cmap=cm.gray_r, extent=(2.3, 3.8, 0.03, 0.45), aspect=1.5 / 0.42)
levels = np.arange(np.amin(lack_of_fit), np.amax(
    lack_of_fit), (np.amax(lack_of_fit) - np.amin(lack_of_fit)) / 20)
# levels = np.hstack(
#     (np.linspace(9, 19, 11), np.linspace(22 ** 0.5, np.sqrt(np.amax(lack_of_fit)), 10) ** 2))
# levels = [5, 6, 7, 8, 9, 11, 13, 15, 17, 20, 24, 29, 35, 41, 47]
print levels
CS = plt.contour(lack_of_fit, levels,
                 origin='lower',
                 cmap=cm.gray,
                 linewidths=1,
                 extent=(2.3, 3.8, 0.03, 0.45))
plt.clabel(CS, levels,  # levels[1::1],
           inline=1,
           fmt='%1.1f',
           fontsize=12)


# plot the crack spacing
csme = [20.1]
cs2 = plt.contour(cs, csme,
                  linewidths=3,
                  colors='k',
                  extent=(2.3, 3.8, 0.03, 0.45), origin='lower')


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


def mean(std, k=50):
    w = np.log(k)
    mu_k = 2.712
    mean = mu_k - std * (-0.007 * w ** 3. + 0.1025 * w ** 2. - 0.8684 * w)
    return mean
mean_arr = mean(m)
plt.plot(mean_arr, m, 'k--')


# CBI = plt.colorbar(im, shrink=0.8)

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


plt.xlabel('mean')
plt.ylabel('stdev')

plt.show()
