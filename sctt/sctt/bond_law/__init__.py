from crack_bridge_models.random_bond_cb import RandomBondCB
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import \
    ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
import matplotlib.pyplot as plt
from spirrid.rv import RV
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

x_arr = np.linspace(-487.40753435, 0., 500)

slip_arr = np.zeros((500, 100))

bond_arr = np.zeros((500, 100))

for k, w in enumerate(np.linspace(1e-5, 1, 100)):

    reinf = ContinuousFibers(r=3.5e-3,
                             tau=RV(
                                 'gamma', loc=0.001260, scale=1.440, shape=0.0539),
                             V_f=0.01,
                             E_f=180e3,
                             xi=fibers_MC(m=6.7, sV0=1.0076),
                             label='carbon',
                             n_int=500)

    ccb = RandomBondCB(E_m=25e3,
                       reinforcement_lst=[reinf],
                       Ll=1500.,
                       Lr=1500.,
                       L_max=400,
                       n_BC=12,
                       w=w)
    ccb.damage
    Kf_intact = ccb.Kf * (1. - ccb.damage)
    mu_T = np.cumsum((ccb.sorted_depsf * Kf_intact)[::-1])[::-1]
    tau = (mu_T / reinf.V_f * reinf.r / 2)[::-1]


#     print len(tau)

    n = (len(ccb._x_arr) + 1) / 2

    epsm = ccb._epsm_arr[1:n]

    epsf_x = np.zeros_like(ccb._x_arr)
    for i, depsf in enumerate(ccb.sorted_depsf):
        epsf_x += np.maximum(ccb._epsf0_arr[i] -
                             depsf * np.abs(ccb._x_arr),
                             ccb._epsm_arr) * ccb.sorted_stats_weights[i]

    epsf = epsf_x[1:n]

    x = ccb._x_arr[1:n]

    slip = cumtrapz(epsf, x) - cumtrapz(epsm, x)

    x1 = x[1::]

    interp_bond = interp1d(x1, tau, bounds_error=False, fill_value=0.)

    interp_slip = interp1d(x1, slip, bounds_error=False, fill_value=0.)

    slip_arr[:, k] = interp_slip(x_arr)

    bond_arr[:, k] = interp_bond(x_arr)

# print ccb._x_arr
print((len(slip_arr[0,:])))

for j in np.hstack((np.arange(10, 499, 50), np.arange(451, 499, 3))):

    plt.plot(slip_arr[j,:], bond_arr[j,:],
             label='z= %0.2f mm' % -x_arr[j])

plt.plot(np.hstack((0, slip_arr[-1,:])), np.hstack((0, bond_arr[-1,:])), label='z=0mm')

plt.xlabel('slip [mm]')
plt.ylabel('bond [MPa]')
plt.legend(loc='best', ncol=2)


# print slip_arr[:, 2]
#
# print bond_arr[:, 2]

plt.show()

# plt.subplot(311)
# plt.plot(slip, tau[::-1])
# plt.title('crack opening=' + str(ccb.w) + 'mm')
# plt.xlabel('slip [mm]')
# plt.ylabel('bond strength [MPa]')
# plt.ylim(0, 0.02)
#
# plt.subplot(312)
# plt.plot(-x[1::], slip)
# plt.ylabel('slip [mm]')
#
#
# plt.subplot(313)
# plt.plot(-x[1::], tau[::-1])
# plt.ylabel('bond strength [MPa]')
# plt.xlabel('x [mm]')
#
# plt.show()

#
#
#
# plt.show()


# print tau
#
#
# tau1 = np.cumsum(
#     (ccb.sorted_depsf * reinf.E_f * reinf.r / 2 * (1. - ccb.damage) * ccb.sorted_stats_weights)[::-1])[::-1]
# print tau1
#
# print tau1 / tau

# z_arr = np.linspace(0, 150, 300)
# Ll_arr = 150. * np.ones_like(z_arr)
# Lr_arr = 5. * np.ones_like(z_arr)
# for load in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
#     sig_m = ccb.get_sig_m_z(z_arr, Ll_arr, Lr_arr, load)
#     plt.plot(z_arr, sig_m, label='load=' + str(load))
# plt.xlim((0, 120))
# plt.legend(loc='best', ncol=2)
# plt.show()
