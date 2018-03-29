'''
Created on 16.11.2017

@author: Yingxiong
'''
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
import matplotlib.pyplot as plt


random_field = RandomField(seed=False,
                           lacor=3.,
                           length=500.,
                           nx=1000,
                           nsim=1,
                           mean=4.5,
                           stdev=0.5,
                           distr_type='Gauss')

x = np.linspace(0, 500, 1000)

plt.plot(x, random_field.random_field)


random_field = RandomField(seed=False,
                           lacor=10.,
                           length=500.,
                           nx=1000,
                           nsim=1,
                           mean=4.5,
                           stdev=0.5,
                           distr_type='Gauss')

plt.plot(x, random_field.random_field)

plt.xlim(0, 500)
plt.ylim(0, 6)
plt.show()