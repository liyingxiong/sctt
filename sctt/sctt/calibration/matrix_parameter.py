'''
Created on Jan 11, 2015

@author: Li Yingxiong
'''
from scipy.special import gamma
import numpy as np

def scale(shape):
    lp = 1.
    lc=1000.
    sig_min=2.73
    f = (lp/(lp+lc))**(1/shape)
    return sig_min/(f*gamma(1+1/shape))

print scale(25.)

from stats.misc.random_field.random_field_1D import RandomField
random_field = RandomField(seed=False,
                       lacor=1.,
                       length=400,
                       nx=1000,
                       nsim=1,
                       loc=.0,
                       shape=25.,
                       scale=scale(25.),
                       distr_type='Weibull')
print np.amin(random_field.random_field)
