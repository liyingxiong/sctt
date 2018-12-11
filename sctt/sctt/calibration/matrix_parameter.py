'''
Created on Jan 11, 2015

@author: Li Yingxiong
'''
from scipy.special import gamma
import numpy as np

s_m = []
for m_m in np.linspace(30., 500., 200):
    s = []
    for sig_min in [2.727, 3.102, 2.927, 3.439]:
        def scale(shape):
            lp = 1.
            lc = 1000.
    #         sig_min= sig_min
            f = (lp / (lp + lc)) ** (1 / shape)
            return sig_min / (f * gamma(1 + 1 / shape))
        s.append(scale(m_m))
    s_m.append(np.mean(s))
print(s_m)
print((np.mean([2.727, 3.102, 2.927, 3.439])))
# print np.mean(s)
# print scale(35.)
# print scale(45.)

# from stats.misc.random_field.random_field_1D import RandomField
# random_field = RandomField(seed=False,
#                        lacor=1.,
#                        length=400,
#                        nx=1000,
#                        nsim=1,
#                        loc=.0,
#                        shape=25.,
#                        scale=scale(25.),
#                        distr_type='Weibull')
# print np.amin(random_field.random_field)
