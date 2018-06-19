'''
Created on 16.05.2018

@author: Yingxiong
'''
from sctt.crack_bridge_models.random_bond_cb import RandomBondCB
import numpy as np
from scipy.interpolate import interp1d
import os.path
from sctt.reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from sctt.composite_tensile_test import CompositeTensileTest
import matplotlib.pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV
from sctt.calibration.matrix_strength_dependence import interp_m_shape

from sctt.sctt_aramis import CTTAramis

reinf = ContinuousFibers(r=3.5e-3,
                         tau=RV(
                             'gamma', loc=0.001260, scale=1.440, shape=0.0539),
                         V_f=0.015,
                         E_f=180e3,
                         xi=fibers_MC(m=6.7, sV0=0.0076),
                         label='carbon',
                         n_int=500)
cb = RandomBondCB(E_m=25e3,
                  reinforcement_lst=[reinf],
                  n_BC=10,
                  L_max=300.)

sig_c_1 = np.array([2.67455075,   2.68525963,   2.71113728,
                    2.77050677,   2.79541756,   3.00115437,   3.07470814,
                    3.34142076,   3.35963343,   3.66234288,   3.92815512,
                    3.97716483,   4.05735791,   4.15105967,   4.22768361,
                    4.2677617,   4.36425778,   4.47337177,   4.73996662,
                    5.04960487,   5.30835567,   5.43752382,   6.04587653,
                    6.41069496,   6.46923713,   6.81507561,   7.38222446,
                    7.95605995,   8.57349024,   9.22669003,   9.26747076,
                    9.39589695,  10.91527796])

crack_position_1 = np.array([374.37437437,  183.18318318,   71.57157157,  274.77477477,
                             469.96996997,   14.01401401,  135.63563564,  427.92792793,
                             322.32232232,  225.22522523,  106.10610611,  296.2962963,
                             352.35235235,  500.,  406.90690691,  161.66166166,
                             248.74874875,   40.04004004,  450.95095095,  120.12012012,
                             88.58858859,  204.2042042,  391.89189189,  335.33533534,
                             53.05305305,  484.48448448,   26.02602603,  259.75975976,
                             150.65065065,  308.30830831,    0.5005005,  363.36336336,
                             171.67167167])

sig_c_15 = np.array([3.64735252,   3.71092609,   3.70825956,
                     3.74009444,   3.88952594,   3.92298145,   4.09889511,
                     4.15996632,   4.35449899,   4.72113307,   4.79430115,
                     5.00599154,   5.30360022,   5.31293615,   5.33435128,
                     5.73145975,   5.79442753,   6.25104164,   6.49266462,
                     6.65478684,   6.7222136,   6.78489012,   6.83373002,
                     6.84497124,   7.24394734,   7.59128258,   7.62672741,
                     7.81204239,   8.15424698,   9.48179584,   9.58694268,
                     10.57443997,  10.8676137,  11.19223141,  11.42821534,
                     11.65163436,  11.70007838,  13.09558499,  13.30071186,
                     13.40929279,  14.53022964,  19.59141555])

crack_position_15 = np.array([292.79279279,  101.6016016,  207.70770771,  500.,
                              18.51851852,  365.86586587,  432.43243243,  151.15115115,
                              254.75475475,  333.83383383,   65.56556557,  462.96296296,
                              132.13213213,  183.68368368,  398.8988989,   49.04904905,
                              231.73173173,  313.31331331,  116.11611612,    0.,
                              415.91591592,  381.88188188,  480.48048048,  275.77577578,
                              170.17017017,   88.58858859,  345.84584585,   36.03603604,
                              446.94694695,  219.71971972,  303.3033033,  243.74374374,
                              196.6966967,   77.07707708,  160.16016016,  266.76676677,
                              490.99099099,  471.47147147,  355.85585586,   27.02702703,
                              142.14214214,    9.50950951])

ctta = CTTAramis(n_x=1000,
                 L=500.,
                 cb=cb,
                 stress=sig_c_15,
                 position=crack_position_15)

sig_c_i, z_x_i, BC_x_i = ctta.gen_data()
load_arr = np.linspace(0, 20., 200)
load_arr = np.unique(np.sort(np.hstack((sig_c_i, load_arr))))
eps_c_arr = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr)
crack_eps_a = ctta.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, sig_c_i)

from scipy.stats import linregress
x = np.linspace(0.002, 0.005, 50)
y = np.interp(x, eps_c_arr, load_arr)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print '1%', slope


plt.plot(eps_c_arr, load_arr)
plt.plot(crack_eps_a, sig_c_i, 'o')

plt.plot([0., 0.007], [0., 0.007 * 1800], 'k--', lw=1)
plt.xlabel('composite strain')
plt.ylabel('composite stress [MPa]')
plt.show()
