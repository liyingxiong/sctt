import numpy as np
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Property, \
    cached_property, Array, Int
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from spirrid.rv import RV
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from scipy.interpolate import interp2d, griddata, LinearNDInterpolator
from scipy.optimize import brentq, fmin
from mpl_toolkits.mplot3d import axes3d
import time as t


class StochasticCB(HasStrictTraits):

    ccb = Instance(CompositeCrackBridge)
    n_BC = Int(8)  # number of different boundary conditions in the interpolator

    n_Int = Property(depends_on='ccb.reinforcement_lst')
    @cached_property
    def _get_n_Int(self):
        n_Int = 0
        for reinf in self.ccb.reinforcement_lst:
            n_Int += reinf.n_int
        return n_Int

    E_m = Property(depends_on='ccb.E_m')
    @cached_property
    def _get_E_m(self):
        return self.ccb.E_m

    E_c = Property(depends_on='ccb.E_c')
    @cached_property
    def _get_E_c(self):
        return self.ccb.E_c

    def minus_sig_c(self, w):  # for evaluation of maximum w
        self.ccb.w = float(w)
        self.ccb.damage
        sig_c = np.sum(self.ccb._epsf0_arr \
                       *self.ccb.sorted_stats_weights \
                       *self.ccb.sorted_V_f * self.ccb.sorted_nu_r \
                       *self.ccb.sorted_E_f * (1. - self.ccb.damage))
        return -sig_c

    ultimate_state = Property(depends_on='n_BC, CB_model')
    @cached_property
    def _get_ultimate_state(self):
        self.ccb.Ll = 1e3
        self.ccb.Lr = 1e3
        w_max = fmin(self.minus_sig_c, 1., full_output=1, disp=0)
        sig_cu = -w_max[1]
        self.ccb.w = w_max[0][0]
        Lmax = self.ccb._x_arr[-2]
        bc_range = np.logspace(np.log10(0.05 * Lmax), np.log10(Lmax), self.n_BC)
        print bc_range
        return bc_range, sig_cu

    BC_range = Property(depends_on='n_BC, CB_model')
    @cached_property
    def _get_BC_range(self):
        return self.ultimate_state[0]

    sig_cu = Property(depends_on='n_BC, CB_model')
    @cached_property
    def _get_sig_cu(self):
        return self.ultimate_state[1]

#     eps_fu = Property(depends_on = 'sig_fu, E_f') # ultimate reinforcement strain
#     @cached_property
#     def _get_eps_fu(self):
#         return self.sig_fu / self.E_f



#=============================================================================
# prepare the interpolator for each Boundary Condition
#=============================================================================
    interps = Property(denpends_on='ccb')
    @cached_property
    def _get_interps(self):
        interps_sigm = []
        interps_epsf = []
        t1 = t.clock()
        print 'preparing the interpolators:'
        for j, L_r in enumerate(self.BC_range):
            for q, L_l in enumerate(self.BC_range):
                if L_l <= L_r:
                    self.ccb.Ll = L_l
                    self.ccb.Lr = L_r
                    w_max = fmin(self.minus_sig_c, 1., disp=0)
                    w_arr = np.linspace(np.sqrt(w_max), 1e-15, 20) ** 2
                    x_arr_record = np.linspace(0, L_r, self.n_Int)
                    epsf_record = []
                    sigm_record = []
                    sigc_record = []
                    for w in w_arr:
                        self.ccb.w = w
                        self.ccb.damage

                        x_arr = self.ccb._x_arr

                        epsf_x = np.zeros_like(x_arr)
                        for i, depsf in enumerate(self.ccb.sorted_depsf):
                            epsf_x += np.maximum(self.ccb._epsf0_arr[i] - \
                                depsf * np.abs(x_arr), \
                                self.ccb._epsm_arr) * self.ccb.sorted_stats_weights[i]

                        sigm_x = self.ccb._epsm_arr * self.ccb.E_m

                        sigc = np.sum(self.ccb._epsf0_arr \
                                    *self.ccb.sorted_stats_weights \
                                    *self.ccb.sorted_V_f * self.ccb.sorted_nu_r \
                                    *self.ccb.sorted_E_f * (1. - self.ccb.damage))

#=============================================================================
# using the LinearNDInterpolator, not efficient
#                         x_arr_record = np.hstack([x_arr_record, x_arr])
#                         sigc_record = np.hstack([sigc_record, sigc*np.ones_like(x_arr)])
#                         epsf_record = np.hstack([epsf_record, epsf_x])
#                         sigm_record = np.hstack([sigm_record, sigm_x])
#
#                     points = np.vstack([x_arr_record, sigc_record]).T
#                     interp_epsf = LinearNDInterpolator(points, epsf_record)
#                     interp_sigm = LinearNDInterpolator(points, sigm_record)
#=============================================================================
#=============================================================================
# reshape the strain and stress array to make the data on a regular grid
                        epsf_x = griddata(x_arr, epsf_x, x_arr_record)
                        sigm_x = griddata(x_arr, sigm_x, x_arr_record)
#=============================================================================

                        epsf_record.append(epsf_x)
                        sigm_record.append(sigm_x)
                        sigc_record.append(sigc)

#=============================================================================
# plot the stress or strain profile under given BC
#                     if L_l == self.BC_range[-1] and L_r == self.BC_range[-1]:
#                         X, Y = np.meshgrid(x_arr_record, sigc_record)
#                         fig = plt.figure()
#                         ax = fig.add_subplot(111, projection='3d')
#                         ax.plot_wireframe(X, Y, sigm_record, rstride=2, cstride=20)
#=============================================================================

                    interp_epsf = interp2d(x_arr_record, sigc_record, epsf_record)
                    interp_sigm = interp2d(x_arr_record, sigc_record, sigm_record)

                    print ((j + 1) * j / 2 + q + 1) * 100 / (self.n_BC * (self.n_BC + 1) / 2), '%'

                    interps_epsf.append(interp_epsf)
                    interps_sigm.append(interp_sigm)

        print 'time consumed:', t.clock() - t1
        return interps_epsf, interps_sigm

#=============================================================================
# functions for evaluation of stress and strain profile
#=============================================================================
    def get_index(self, Ll, Lr):
        # find the index of the interpolator corresponding to the BC
        l, r = np.sort([Ll, Lr])
        i = (np.abs(self.BC_range - l)).argmin()
        j = (np.abs(self.BC_range - r)).argmin()
        return (j + 1) * j / 2 + i

    # function for evaluating specimen reinforcement strain
    def get_eps_f_z(self, z_arr, Ll_arr, Lr_arr, load):
        def get_eps_f_i(self, z, Ll, Lr, load):
            ind = self.get_index(Ll, Lr)
            f = self.interps[0][ind]
            return f(z, load)
        v = np.vectorize(get_eps_f_i)
        return v(self, z_arr, Ll_arr, Lr_arr, load)


    # function for evaluating specimen matrix stress
    def get_sig_m_z(self, z_arr, Ll_arr, Lr_arr, load):
        def get_sig_m_i(self, z, Ll, Lr, load):
            ind = self.get_index(Ll, Lr)
            f = self.interps[1][ind]
            return f(z, load)
        v = np.vectorize(get_sig_m_i)
        return v(self, z_arr, Ll_arr, Lr_arr, load)

if __name__ == '__main__':

    from matplotlib import pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     from matplotlib import cm
#     import time as t

    reinf = ContinuousFibers(r=0.0035,
                          tau=RV('weibull_min', loc=0.0, shape=1., scale=1.),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=2.0, sV0=0.003),
                          label='carbon',
                          n_int=100)

    cb1 = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf]
                                 )

    scb = StochasticCB(ccb=cb1)

#     print scb.data[0].shape
#     print scb.data[1].shape

    z = np.linspace(0, 500, 501)
#     _lambda = np.ones_like(x)*60
    LL = np.ones_like(z) * 1
    LR = np.ones_like(z) * 90
#
#     BC = np.vstack([LL, LR])

#     print scb.get_index(90, 90)


#     sig_m = scb.get_sig_m_z(points.T)
#     t2 = t.clock()
#     print '--', t2
#     for i in range(1000):
    sig_m = scb.get_sig_m_z(z, LL, LR, 100)
#     print 't2', t.clock()-t2
#     print x, sig_m
#     sig_m = scb.get_sig_m_z(points.T)
#     x = scb.data[0][0, :]
#     y = scb.data[1]
#
#     print len(x)
#
    fig1 = plt.figure()
    ax2 = fig1.add_subplot(111)
    ax2.plot(z, sig_m)
    plt.show()


