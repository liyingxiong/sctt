'''
Created on Jun 21, 2018

@author: liyin
'''

import matplotlib.pyplot as plt
from spirrid.rv import RV
import numpy as np

tau=RV('gamma', loc=0.001260, scale=1.440, shape=0.0539)

tau_1=RV('beta', loc=0.001260, scale=1.440, shape=0.0539)


x = np.linspace(0, 10, 1000)
y = tau_1.pdf(x)

plt.plot(x,y)
plt.show()