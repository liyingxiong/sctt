'''
Created on Dec 11, 2014

@author: Li Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

a = np.array([1, 3, 3, 4])
# a[:] = np.array([4, 5, 6])
# print a
c = np.array([1,2,4, 5])
# a = np.ones(1)

b = np.array([1, -1, -1, 1, -1])


d = csr_matrix(np.vstack((a, c)))
               
e = np.array([1, 1])

print type(d.T.dot(e))
                