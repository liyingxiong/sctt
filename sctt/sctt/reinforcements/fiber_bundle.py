from traits.api import HasTraits, Array, Instance, List, Float, Int, \
    Property, cached_property

import numpy as np
from spirrid.rv import RV
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers
from util.traits.either_type import EitherType


class FiberBundle(HasTraits):

    #=========================================================================
    # Parameters
    #=========================================================================
    E_f = Float(180e3)  # the elastic modulus of the fiber
    V_f = Float  # volume fraction
# tau = EitherType(klasses=[FloatType, Array, RV]) # bond stiffness
    tau = Array
    tau_weights = Array  # the weights for tau, applicable when tau is an array
    xi = EitherType(klasses=[float, RV, WeibullFibers])  # breaking strain
# r = EitherType(klasses=[FloatType, RV]) # fiber radius
    r = Float  # fiber radius
    n_int = Int(10)  # number of integration points

    #=========================================================================
    # Sampling
    #=========================================================================
    samples = Property(depends_on='r, V_f, E_f, xi, tau, n_int')

    @cached_property
    def _get_samples(self):
        if isinstance(self.tau, np.ndarray):
            tau_arr = self.tau
            stat_weights = self.tau_weights
        return 2 * tau_arr / self.r / self.E_f, stat_weights, \
            np.ones_like(tau_arr), self.r * np.ones_like(tau_arr)

    depsf_arr = Property(depends_on='r, V_f, E_f, xi, tau, n_int')

    @cached_property
    def _get_depsf_arr(self):
        return self.samples[0]

    stat_weights = Property(depends_on='r, V_f, E_f, xi, tau, n_int')

    @cached_property
    def _get_stat_weights(self):
        return self.samples[1]

    nu_r = Property(depends_on='r, V_f, E_f, xi, tau, n_int')

    @cached_property
    def _get_nu_r(self):
        return self.samples[2]

    r_arr = Property(depends_on='r, V_f, E_f, xi, tau, n_int')

    @cached_property
    def _get_r_arr(self):
        return self.samples[3]


if __name__ == '__main__':

    reinf = FiberBundle(r=0.0035,
                        tau=np.array([0.02680217,  0.03862634,  0.04707699,  0.0542814,  0.06082189, 0.06697246,  0.07289507,  0.07870153,  0.08447959,  0.0903067,
                                      0.09625907,  0.10241951,  0.10888651,  0.11578704,  0.1232982, 0.13168872,  0.14141016,  0.15333842,  0.16964278,  0.20090774]),
                        tau_weights=np.array(
                            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
                        V_f=0.1,
                        E_f=200e3,
                        xi=0.035)
    print((reinf.depsf_arr))
