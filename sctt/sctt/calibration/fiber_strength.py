from scipy.special import gamma
from scipy.constants import pi
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


def strength(m, sV0, L):
    r = 3.5e-3
    E = 181e3
    return (pi * r ** 2 * L) ** (-1 / m) * sV0 * gamma(1 + 1 / m) * E


def bundle_strength(m, sV0, L):
    r = 3.5e-3
    E = 181e3
    return E * sV0 * (m * L * pi * r ** 2) ** (-1 / m) * np.exp(-1 / m)

print((strength(8.806, 0.0134, 100)))

# s = []
# m_arr = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
# sV0_arr = np.array([0.0070, 0.0075, 0.0080, 0.0085, 0.0090, 0.0095])
# for m in m_arr:
#     for sV0 in sV0_arr:
#         s.append(strength(m, sV0, 100))
#
# print s
# s_arr = np.array(s).reshape(6, 6)
#
#
# x, y = np.meshgrid(sV0_arr, m_arr)
#
# print s_arr
# print x
# print y
#
# outfile = 'D:\\1.txt'
# np.savetxt(outfile, s_arr, fmt='%.4f', delimiter=',')
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(x, y, s_arr, rstride=1, cstride=1)
# ax.set_xlabel('sV0')
# ax.set_ylabel('M')
# ax.set_zlabel('theoretical bundle strength [Mpa]')
#
# plt.show()
