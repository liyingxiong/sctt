'''
Created on 16.04.2016

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.widgets import CheckButtons

fig, ax = plt.subplots()

fpath = 'D:\\data\\Tensile_test_multiple_cracking\\ss_curve_mm_24_1\\'
s_m_arr = np.linspace(2.4, 3.8, 40)

label_position = plt.axes([0.05, 0.1, 0.2, 0.8])
d_set = {}

# plot simulation
for s_m in s_m_arr:
    stress, strain = np.loadtxt(fpath + str(s_m) + '.txt')
    flabel = 's_m=' + str(s_m)
    d_set[flabel], = ax.plot(strain, stress, visible=False)

# plot experiments
for i in np.arange(1, 6):
    filepath1 = 'D:\\data\\Tensile_test_multiple_cracking\\TT-4C-0' + \
        str(i) + '.txt'
    data = np.loadtxt(filepath1, delimiter=';')
    eps = -data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.
    sig = data[:, 1] / 2.
    flabel = 'experiment-' + str(i)
    d_set[flabel], = ax.plot(eps, sig, color='0.7', visible=False)

ax.set_xlabel('strain')
ax.set_ylabel('stress [MPa]')


plt.subplots_adjust(left=0.32, right=0.99)

check = CheckButtons(
    label_position, sorted(d_set.keys()), np.zeros(len(d_set.keys())))


def func(label):
    d_set[label].set_visible(not d_set[label].get_visible())
    plt.draw()

check.on_clicked(func)


plt.show()
