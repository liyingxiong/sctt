from sctt_aramis import CTTAramis
import numpy as np
from matplotlib import pyplot as plt
from tau_strength_dependence import interp_tau_shape, interp_tau_scale
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from crack_bridge_models.random_bond_cb import RandomBondCB
import os
from scipy.interpolate import interp1d
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from scipy.stats import variation


class prettyfloat(float):

    def __repr__(self):
        return "%0.4f" % self

# homedir = 'D:\\data\\'
# path = [homedir, 'test5.txt']
# filepath = os.path.join(*path)
# data = np.loadtxt(filepath)
#
# stress = data[3, :]/2.
# time = data[2, :]
# position=data[1, :]
#
# index = np.argsort(time)
# stress = stress[index]
# position = position[index]


# m = 5
# sV0 = 0.007094400237837161

# m = 11.
# sV0 = 0.010372329629894428

m = 10.
sV0 = 0.0095

# sV0_arr = np.array([0.008043056417130334, 0.008689444790342452, 0.009133666555177156, 0.00954231413126173, 0.009981034909603366, 0.010372329629894428])
# m_arr = np.array([6., 7., 8., 9., 10., 11.])


# shape = float(interp_tau_shape(sV0, m))
# scale = float(interp_tau_scale(sV0, m))
#
# print shape, scale

#
# print shape, scale

# shape = 0.0479314809805*1.0
# scale = 2.32739332143*1.0

shape = 0.0479314809805 * 1.0
scale = 2.32739332143


reinf1 = ContinuousFibers(r=3.5e-3,
                          tau=RV('gamma', loc=0., scale=scale, shape=shape),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=m, sV0=sV0),
                          label='carbon',
                          n_int=500)

cb = RandomBondCB(E_m=25e3,
                  reinforcement_lst=[reinf1],
                  n_BC=8,
                  L_max=120)


def get_z(y, position):
    try:
        z_grid = np.abs(position - y)
        return np.amin(z_grid)
    except ValueError:  # no cracks exist
        return np.ones_like(position) * 2. * 1000.


def get_BC(y, position, L):
    try:
        y = np.sort(y)
        d = (y[1:] - y[:-1]) / 2.0
        # construct a piecewise function for interpolation
        xp = np.hstack([0, y[:-1] + d, L])
        L_left = np.hstack([y[0], d, np.NAN])
        L_right = np.hstack([d, L - y[-1], np.NAN])
        f = interp1d(xp, np.vstack([L_left, L_right]), kind='zero')
        return f(position)
    except IndexError:
        #         print y
        return np.vstack([np.zeros_like(position), np.zeros_like(position)])


def matrix_stress(position, stress):
    #     print stress
    sig_m_arr = []
    for i, sig_c in enumerate(stress):
        y = position[:i]
        z = get_z(y, position[i])
#         print sig_c, z
        Ll, Lr = get_BC(y, position[i], 120.)
#         print 'BC;', Ll, Lr
        if Ll == 0.:
            sig_m_arr.append(
                cb.E_m * sig_c / ((reinf1.V_f * reinf1.E_f) + cb.E_m * (1 - reinf1.V_f)))
        else:
            sig_m = cb.get_sig_m_z(z, Ll, Lr, sig_c)
            sig_m_arr.append(float(sig_m))
    return sig_m_arr

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


stress4 = np.array([])
rec_4 = np.array([])
for i in range(4):
    # for i in [3]:
    homedir = 'D:\\data\\'
    path = [homedir, 'test' + str(i + 1) + '.txt']
    filepath = os.path.join(*path)
    data = np.loadtxt(filepath, delimiter=',')

    position = data[0, :]
    stress = data[1, :] / 2.
#     rec_4 = np.hstack((rec_4, stress))
    #
    index = np.argsort(stress)
    stress = stress[index]
    position = position[index]


#     print 'position', map(prettyfloat, position.tolist())
#     print 'force[KN]', map(prettyfloat, (2.*stress).tolist())
#     print 'matrix stress', map(prettyfloat, matrix_stress(position, stress))
#     print ' '

#     ctta = CTTAramis(n_x=400,
#              L=120,
#              cb=cb,
#              stress = stress,
#              position=position)
#
#     sig_c_i, z_x_i, BC_x_i = ctta.gen_data()
#     load_arr = np.linspace(0, 12., 200)
#     load_arr = np.sort(np.hstack((sig_c_i, load_arr)))
#     eps_c_arr = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
#     crack_eps_a = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)
#
#     plt.plot(eps_c_arr, load_arr, 'k', label=str(i+1))
#     plt.plot(crack_eps_a, sig_c_i, 'k.')

# plt.legend()
# plt.show()


reinf2 = ContinuousFibers(r=3.5e-3,
                          tau=RV('gamma', loc=0., scale=scale, shape=shape),
                          V_f=0.015,
                          E_f=180e3,
                          xi=fibers_MC(m=m, sV0=sV0),
                          label='carbon',
                          n_int=500)

cb2 = RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf2],
                   n_BC=8,
                   L_max=120)


def get_z1(y, position):
    try:
        z_grid = np.abs(position - y)
        return np.amin(z_grid)
    except ValueError:  # no cracks exist
        return np.ones_like(position) * 2. * 1000.


def get_BC1(y, position, L):
    try:
        y = np.sort(y)
        d = (y[1:] - y[:-1]) / 2.0
        # construct a piecewise function for interpolation
        xp = np.hstack([0, y[:-1] + d, L])
        L_left = np.hstack([y[0], d, np.NAN])
        L_right = np.hstack([d, L - y[-1], np.NAN])
        f = interp1d(xp, np.vstack([L_left, L_right]), kind='zero')
        return f(position)
    except IndexError:
        #         print y
        return np.vstack([np.zeros_like(position), np.zeros_like(position)])


def matrix_stress1(position, stress):
    #     print stress
    sig_m_arr = []
    for i, sig_c in enumerate(stress):
        y = position[:i]
        z = get_z1(y, position[i])
#         print sig_c, z
        Ll, Lr = get_BC1(y, position[i], 120.)
#         print 'BC;', Ll, Lr
        if Ll == 0.:
            sig_m_arr.append(
                cb2.E_m * sig_c / ((reinf1.V_f * reinf1.E_f) + cb2.E_m * (1 - reinf1.V_f)))
        else:
            sig_m = cb2.get_sig_m_z(z, Ll, Lr, sig_c)
            sig_m_arr.append(float(sig_m))
    return sig_m_arr

stress6 = np.array([])
rec_6 = np.array([])
#
for i in range(4):
    # for i in [1]:
    homedir = 'D:\\data\\'
    path = [homedir, 'test6' + str(i + 1) + '.txt']
    filepath = os.path.join(*path)
    data = np.loadtxt(filepath, delimiter=',')

    position = data[0, :]
    stress = data[1, :] / 2.

#     ctta1 = CTTAramis(n_x=400,
#          L=120,
#          cb=cb2,
#          stress = stress,
#          position=position)
#
#     sig_c_i, z_x_i, BC_x_i = ctta1.gen_data()
#     load_arr1 = np.linspace(0, 20., 200)
#     load_arr1 = np.sort(np.hstack((sig_c_i, load_arr1)))
#     eps_c_arr1 = ctta1.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr1)
#     crack_eps_a = ctta1.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)
#
#     plt.plot(eps_c_arr1, load_arr1, 'k--', label=str(1.5))
#     plt.plot(crack_eps_a, sig_c_i, 'k.')
#
# home_dir = 'D:\\Eclipse\\'
# for i in range(5):
# path1 = [home_dir, 'git',  # the path of the data file
# 'rostar',
# 'scratch',
# 'diss_figs',
# 'TT-4C-0'+str(3+1)+'.txt']
# filepath1 = filepath = os.path.join(*path1)
#
# path2 = [home_dir, 'git',  # the path of the data file
# 'rostar',
# 'scratch',
# 'diss_figs',
# 'TT-6C-0'+str(1+1)+'.txt']
# filepath2 = os.path.join(*path2)
#
# data = np.loadtxt(filepath1, delimiter=';')
# plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='0.5', label='experiment')
# data = np.loadtxt(filepath2, delimiter=';')
# plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='0.5')
#
# plt.legend(loc='best')
#
#
# plt.show()
#     rec_6 = np.hstack((rec_6, stress))
#     #
    index = np.argsort(stress)
    stress = stress[index]
    position = position[index]

    if i == 0:
        plt.plot(np.hstack((0., stress, 21.00)), np.hstack((0, np.arange(
            len(stress) + 1) / 120.)), color='0.5', drawstyle='steps', label='experiment')
    else:
        plt.plot(np.hstack((0., stress, 21.00)), np.hstack((0, np.arange(
            len(stress) + 1) / 120.)), color='0.5', drawstyle='steps')

stress_1 = np.array([4.01595859,   4.80487451,   5.53772944,
                     6.46435667,   6.52607575,   8.11517871,   8.6199053,
                     9.62836115,  10.66232725,  13.19993526,  14.50352367,
                     15.72521958,  18.22227039,  19.39226057])

plt.plot(np.hstack((0., stress_1, 21.00)), np.hstack((0, np.arange(
    len(stress_1) + 1) / 120.)), 'k', lw=2, drawstyle='steps', label='crack tracing algorithm')


#     stress4 = np.hstack((stress4, matrix_stress(position, stress)))
plt.xlabel('stress [MPa]')
plt.ylabel('crack density [1/mm]')
plt.legend(loc='best')
plt.show()

#
#     stress6 = np.hstack((stress6, matrix_stress1(position, stress)))
#
#     print 'position', map(prettyfloat, position.tolist())
#     print 'force[KN]', map(prettyfloat, (2. * stress).tolist())
#     print 'matrix stress', map(prettyfloat, matrix_stress1(position, stress))
#     print ' '
#
#
#
#
bin_arr = np.linspace(0.0, 6., 25)

# hist, bins = np.histogram(stress4_arr, bins=bin_arr)
plt.hist(stress4, bin_arr,  color='0.3', label='1')
print 'avg', np.average(stress4)
print 'COV', variation(stress4)
# plt.hist(rec_4*(cb.E_m/cb.E_c), bin_arr, label='mixture rule')
plt.xlabel('matrix stress[MPa]')
plt.legend()
plt.figure()
plt.hist(stress6, bin_arr,  color='0.5', alpha=0.7, label='1.5')
print 'avg', np.average(stress6)
print 'COV', variation(stress6)
# plt.hist(rec_6*(cb2.E_m/cb2.E_c), bin_arr, label='mixture rule')
plt.xlabel('matrix stress[MPa]')
# plt.legend()
plt.show()


# stress6 = []
#
# for i in range(4):
#     homedir = 'D:\\data\\'
#     path = [homedir, 'test'+str(5)+'.txt']
#     filepath = os.path.join(*path)
#     data = np.loadtxt(filepath, delimiter=',')
#
#     position=data[0, :]
#     stress = data[1, :]/2.
#     #
#     index = np.argsort(stress)
#     stress = stress[index]
#     position = position[index]
#
#     stress4.append( matrix_stress(position, stress) )
