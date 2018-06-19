'''
Created on May 27, 2015

@author: Yingxiong
'''
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
from matplotlib import pyplot as plt

# var = []
# for m in np.linspace(10, 200, 50):
#     print m
#     var1 = 0
#     for i in range(5):
random_field = RandomField(seed=False,
                           lacor=10.,
                           length=500.,
                           nx=500,
                           nsim=1,
                           loc=.0,
                           scale=3.2,
                           shape=50,
                           distr_type='Weibull')
#         var1 += 0.2 * \
#             (np.amax(random_field.random_field) -
#              np.amin(random_field.random_field)) / 3.2
#     var.append(var1)
plt.ylim((2.5, 3.5))
plt.plot(random_field.xgrid, random_field.random_field)
plt.plot((0, 500), (3.2, 3.2), 'k--')
plt.show()
