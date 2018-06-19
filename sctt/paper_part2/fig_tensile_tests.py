'''
Created on 06.11.2017

@author: Yingxiong
'''
import numpy as np
import os.path
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

# plot the experimental sig_eps responses
home_dir = 'D:\\Eclipse\\'
# for i in range(5):
for i in range(5):
    path1 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-4C-0' + str(i + 1) + '.txt']
    filepath1 = filepath = os.path.join(*path1)

    path2 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-6C-0' + str(i + 1) + '.txt']
    filepath2 = os.path.join(*path2)

    data = np.loadtxt(filepath1, delimiter=';')
    ax2.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.3')

    data = np.loadtxt(filepath2, delimiter=';')
    ax2.plot(-data[:, 2] / 2. / 250. - data[:, 3] / 2. / 250.,
             data[:, 1] / 2., lw=1, color='0.8')


# =========================================================================
# 1% aramis
s1t = np.array([10.00872464,   5.52844059,  11.25707195,
                7.12078782,   7.5985656,   7.12078782,  12.27597275])
s2t = np.array([17.1316769,   9.72990228,  6.97622649,   8.94871857])
s3t = np.array(
    [4.40479296,   7.95527934,  17.34835118,   8.85553078,   6.56429542])
s4t = np.array(
    [15.18919693,   7.0975642,   7.97860545,  11.93039465,   9.31819916])
s5t = np.array([21.13306692,   8.62724404,  18.52434985,
                14.0797884,   7.63826555,   8.38268409,  21.98885654])

s1b = np.array(
    [6.02457869, 22.90243875,  11.25707195,   7.35146734,   8.93051279,  16.18415792])
s2b = np.array(
    [7.10445464,   9.2492387,   6.97622649,  16.79998289, 8.77070183,   8.77070183])
s3b = np.array(
    [4.40479296,  14.90546437,   5.49085621,   4.40479296, 5.97395657,  17.01455883])
s4b = np.array([19.68039936,   6.88058764,  13.92738198,
                7.0975642, 7.0975642,   7.97860545,  17.20897856])
s5b = np.array([22.92756902,   8.62724404,   8.99976055,
                10.78376281, 7.17470687,  12.65419438,   8.23534141,   5.11059319])
strength_e = [12.5, 10.38, 11.34, 12.43, 13.40]
# =========================================================================

#=========================================================================
#1.5% aramis

s1t5 = np.array([16.91681074,  28.24358356,  14.16653556,   7.85852962,
                 9.26719295,  35.72474096,  15.07669087,  11.44987103,
                 9.26719295,   9.89696334,   9.58805621])
s2t5 = np.array([14.03091356,  18.79005001,  13.87798694,  15.37755049,
                 17.00879121,  15.98687316,  15.59358517,  12.76183435])
s3t5 = np.array([9.05175008,  10.76772668,   9.05175008,   9.05175008,
                 19.71997199,  11.42490661,  11.21500315,  12.15511804,
                 7.663163,  18.42709064,   6.37574066,  15.91778204])
s4t5 = np.array([10.36301934,  18.73350336,  10.00397938,   7.78785911,
                 11.22337517,  10.72917466,  11.89792408,  17.01613892,  10.86517866])
s5t5 = np.array([18.0952497,    0.60585937,
                 19.90342057,   0.60585937,  31.91502337, 11.28946654,  11.71460843,   0.60585937,  12.16635002])

s1b5 = np.array([11.85898609,  19.53944724,
                 11.44987103,  11.71533759,  14.91634299,  34.56317101,
                 10.66047702,   7.48661655,  40.40901103])
s2b5 = np.array([10.48396872,  25.14811366,  12.08379738,  11.80306615,
                 11.3517029,  10.27731759,  14.15291021,  32.14101299,  10.74935872])

s3b5 = np.array([16.02677067,   7.47010424,   8.7946856,  13.85198624,
                 7.663163,   4.19339853,  26.06350928,   7.47010424,
                 8.25237909,   7.25361537])
s4b5 = np.array([16.51271353,  13.22650031,  37.56411316,   9.14751061,
                 32.29994302,  18.90466892,   7.54397453,   9.34653494])

s5b5 = np.array([10.81934181,   0.60585937,  35.98725918,  10.65285666,   0.60585937,   0.6058362,
                 0.60549606,  15.01184154,   0.60585937])
strength_e5 = [20.66, 21.29, 20.76, 20.63, 19.60]
#=========================================================================
s1e = np.sort(np.hstack((s1t, s1b)))
s2e = np.sort(np.hstack((s2t, s2b)))
s3e = np.sort(np.hstack((s3t, s3b)))
s4e = np.sort(np.hstack((s4t, s4b)))
s5e = np.sort(np.hstack((s5t, s5b)))
s_lste = [s1e, s2e, s3e, s4e, s5e]

s1e5 = np.sort(np.hstack((s1t5, s1b5)))
s2e5 = np.sort(np.hstack((s2t5, s2b5)))
s3e5 = np.sort(np.hstack((s3t5, s3b5)))
s4e5 = np.sort(np.hstack((s4t5, s4b5)))
s5e5 = np.sort(np.hstack((s5t5, s5b5)))
s_lste5 = [s1e5, s2e5, s3e5, s4e5, s5e5]

lines = []


for i in range(4):

    path1 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-4C-0' + str(i + 1) + '.txt']
    filepath1 = filepath = os.path.join(*path1)

    data = np.loadtxt(filepath1, delimiter=';')
    experi_eps, experi_sig = -data[:, 2] / 2. / \
        250. - data[:, 3] / 2. / 250., data[:, 1] / 2.

    a = s_lste[i]
    nc = np.arange(len(a))
    cs = 240. / (nc + 1)
    cs[cs >= 120] = 120

    strain = np.interp(
        np.hstack((0, a / 2, strength_e[i])), experi_sig, experi_eps)
    ax1.step(strain, np.hstack((120, cs, cs[-1])), color='0.3')

    path2 = [home_dir, 'git',  # the path of the data file
             'rostar',
             'scratch',
             'diss_figs',
             'TT-6C-0' + str(i + 1) + '.txt']
    filepath2 = filepath = os.path.join(*path2)

    data = np.loadtxt(filepath2, delimiter=';')
    experi_eps, experi_sig = -data[:, 2] / 2. / \
        250. - data[:, 3] / 2. / 250., data[:, 1] / 2.

    a = s_lste5[i]
    nc = np.arange(len(a))
    cs = 240. / (nc + 1)
    cs[cs >= 120] = 120

    strain = np.interp(
        np.hstack((0, a / 2, strength_e5[i])), experi_sig, experi_eps)
    l, = ax1.step(strain, np.hstack((120, cs, cs[-1])), color='0.8')
    lines.append(copy)


plt.show()
