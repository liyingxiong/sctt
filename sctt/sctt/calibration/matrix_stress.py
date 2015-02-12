from sctt_aramis import CTTAramis
import numpy as np
from matplotlib import pyplot as plt
from calibration.tau_strength_dependence import interp_tau_shape, interp_tau_scale
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from crack_bridge_models.random_bond_cb import RandomBondCB
import os
from scipy.interpolate import interp1d
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV


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

m = 11.
sV0 = 0.010372329629894428

# sV0_arr = np.array([0.008043056417130334, 0.008689444790342452, 0.009133666555177156, 0.00954231413126173, 0.009981034909603366, 0.010372329629894428])
# m_arr = np.array([6., 7., 8., 9., 10., 11.])


shape = float(interp_tau_shape(sV0, m))
scale = float(interp_tau_scale(sV0, m))

# shape = 0.057406221892621546
# scale = 5.4207851813009602

reinf1 = ContinuousFibers(r=3.5e-3,
                      tau=RV('gamma', loc=0., scale=scale, shape=shape),
                      V_f=0.015,
                      E_f=180e3,
                      xi=fibers_MC(m=m, sV0=sV0),
                      label='carbon',
                      n_int=500)

cb =  RandomBondCB(E_m=25e3,
                   reinforcement_lst=[reinf1],
                   n_BC = 12,
                   L_max = 120)


def get_z(y, position):
    try:
        z_grid = np.abs(position - y)
        return np.amin(z_grid)
    except ValueError: #no cracks exist
        return np.ones_like(position)*2.*1000.
    
def get_BC(y, position, L):
    try:
        y = np.sort(y)
        d = (y[1:] - y[:-1]) / 2.0
        #construct a piecewise function for interpolation
        xp = np.hstack([0, y[:-1]+d, L])
        L_left = np.hstack([y[0], d, np.NAN])
        L_right = np.hstack([d, L-y[-1], np.NAN])
        f = interp1d(xp, np.vstack([L_left, L_right]), kind='zero')
        return f(position)
    except IndexError:
        print y
        return np.vstack([np.zeros_like(position), np.zeros_like(position)])

def matrix_stress(position, stress):
    print stress
    sig_m_arr = []
    for i, sig_c in enumerate(stress):
        y = position[:i]
        z = get_z(y, position[i])
        print sig_c, z
        Ll, Lr = get_BC(y, position[i], 120.)
        print 'BC;', Ll, Lr
        if Ll == 0.:
            sig_m_arr.append(cb.E_m*sig_c/((reinf1.V_f*reinf1.E_f)+cb.E_m*(1-reinf1.V_f)))
        else:
            sig_m = cb.get_sig_m_z(z, Ll, Lr, sig_c)
            sig_m_arr.append(float(sig_m))
    return sig_m_arr

# print get_BC(position[:7], position[7], 120.)
# print position
# print position[:4], position[4]

# for i in range(5):
homedir = 'D:\\data\\'
path = [homedir, 'test'+str(61)+'.txt']
filepath = os.path.join(*path)
data = np.loadtxt(filepath)

stress = data[3, :]/2.
time = data[2, :]
position=data[1, :]

index = np.argsort(time)
stress = stress[index]
position = position[index]

print matrix_stress(position, stress)
print position, stress




