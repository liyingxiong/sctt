'''
Created on Dec 11, 2014

@author: Li Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


tau_shape = [0.079392235619918011, 0.070557619484416842, 0.063299300020833421, 0.057273453501868139, 0.052218314012133477, 0.04793148098051675, 0.10184138982654255, 0.090386833801161789, 0.081105577459563358, 0.073447931456124799, 0.067031491579850458, 0.061583354951796301, 0.12549447436183159, 0.11100174486617211, 0.099332453863924197, 0.089751925690600545, 0.081757089644320463, 0.074992574602765982, 0.14872431076672932, 0.1310279952657768, 0.11687910678733873, 0.10533009089253634, 0.09573957473543436, 0.08765870476491136, 0.17114399856895063, 0.15019973334235998, 0.13356519781922624, 0.12006215401696788, 0.10890090044135943, 0.099533619810760268, 0.19267340894906607, 0.16847672735868743, 0.14937828293751562, 0.1339549206939461, 0.12126139030716455, 0.11064685293542306]
tau_scale = [0.85377504364710732, 1.0754895775375046, 1.3336402595345596, 1.6285211551178209, 1.9598281655654299, 2.3273933214348754, 0.56417269946935489, 0.70916030125575424, 0.87406020942523077, 1.0594252661569492, 1.2658121316937005, 1.4937628940125933, 0.39043238917402051, 0.49222068984548584, 0.60811648588336797, 0.7385866765422433, 0.8840641933601644, 1.0449527228922362, 0.28950334532469602, 0.36673311911773132, 0.45488021504930615, 0.55431262938885573, 0.6653642278483789, 0.78834363676500241, 0.22661952350942449, 0.28843443700068938, 0.35916575964517833, 0.43910806699061661, 0.52853103472944241, 0.62768076866581679, 0.184677601592135, 0.23610461900819166, 0.29508421433871257, 0.36185689380081015, 0.4366456989944002, 0.5196643843086165]


# sm=3.3788, psi=35
# n_cracks_1 = np.array([20, 25, 25, 26, 30, 32, 17, 19, 23, 25, 26, 26, 16, 17, 17, 22, 25, 25, 15, 17, 17, 17, 21, 23, 14, 16, 17, 17, 17, 19, 13, 15, 16, 17, 17, 17])
# n_cracks_1_5 = np.array([26, 30, 35, 37, 39, 44, 25, 26, 28, 32, 36, 37, 22, 25, 26, 27, 31, 33, 19, 22, 25, 26, 26, 29, 18, 19, 22, 25, 26, 26, 17, 18, 20, 22, 25, 26])

# sm=3.2228, psi=45
# n_cracks_1 = np.array([22, 24, 27, 28, 34, 38, 19, 22, 23, 26, 28, 29, 16, 19, 22, 22, 24, 26, 16, 16, 19, 21, 22, 23, 14, 16, 16, 19, 20, 22, 12, 16, 16, 16, 19, 21])
# n_cracks_1_5 = np.array([28, 32, 36, 38, 42, 49, 26, 28, 32, 36, 37, 39, 24, 26, 27, 29, 33, 36, 18, 24, 26, 27, 29, 32, 18, 20, 25, 26, 27, 29, 16, 18, 23, 25, 26, 27])

# sm=3.6783, psi=25
n_cracks_1=np.array([19, 20, 23, 24, 27, 28, 17, 18, 20, 23, 23, 25, 16, 17, 19, 20, 21, 23, 16, 16, 16, 18, 19, 20, 14, 16, 16, 16, 18, 18, 13, 14, 16, 16, 16, 18])
n_cracks_1_5= np.array([24, 26, 31, 33, 37, 38, 23, 24, 26, 29, 31, 34, 20, 22, 23, 26, 29, 31, 17, 20, 21, 23, 26, 26, 17, 17, 20, 21, 23, 24, 14, 17, 18, 21, 21, 23])

m_arr = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
sV0_arr = np.array([0.0070, 0.0075, 0.0080, 0.0085, 0.0090, 0.0095])
x, y =np.meshgrid(sV0_arr, m_arr)

n_cracks_1 = n_cracks_1.reshape(6,6)
n_cracks_1_5 = n_cracks_1_5.reshape(6,6)

# print n_cracks_1/400.
# print n_cracks_1_5/400.

tau_shape = np.array(tau_shape).reshape(6,6)
tau_scale = np.array(tau_scale).reshape(6,6)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')    

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot_surface(x, y, n_cracks_1_5/400., rstride=1, cstride=1)
ax.plot_surface(x, y, tau_scale, rstride=1, cstride=1)
# ax.plot_wireframe(x, y, expri, color="red")
ax.set_xlabel('$s_\mathrm{f}$')
ax.set_ylabel('$\psi_\mathrm{f}$')
# ax.set_zlabel('crack density [1/mm]')
ax.set_zlabel('$s_\tau$')
# ax.set_zlim(0.0, 0.15)
plt.show()
                