'''
Created on 09.10.2014

@author: Li Yingxiong
'''

from crack_bridge_models.random_bond_cb import RandomBondCB
from calibration import Calibration
import numpy as np
from scipy.interpolate import interp1d
import os.path
from reinforcements.fiber_bundle import FiberBundle
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from composite_tensile_test import CompositeTensileTest
import matplotlib.pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from spirrid.rv import RV



random_field = RandomField(seed=False,
                       lacor=1.,
                       length=400,
                       nx=1000,
                       nsim=1,
                       loc=.0,
                       shape=25.,
                       scale=1.3*3.67829544828,
                       distr_type='Weibull')

tau_shape = np.array([0.079392235619918011, 0.070557619484416842, 0.063299300020833421, 0.057273453501868139, 0.052218314012133477, 0.04793148098051675, 0.10184138982654255, 0.090386833801161789, 0.081105577459563358, 0.073447931456124799, 0.067031491579850458, 0.061583354951796301, 0.12549447436183159, 0.11100174486617211, 0.099332453863924197, 0.089751925690600545, 0.081757089644320463, 0.074992574602765982, 0.14872431076672932, 0.1310279952657768, 0.11687910678733873, 0.10533009089253634, 0.09573957473543436, 0.08765870476491136, 0.17114399856895063, 0.15019973334235998, 0.13356519781922624, 0.12006215401696788, 0.10890090044135943, 0.099533619810760268, 0.19267340894906607, 0.16847672735868743, 0.14937828293751562, 0.1339549206939461, 0.12126139030716455, 0.11064685293542306])
tau_scale = np.array([0.85377504364710732, 1.0754895775375046, 1.3336402595345596, 1.6285211551178209, 1.9598281655654299, 2.3273933214348754, 0.56417269946935489, 0.70916030125575424, 0.87406020942523077, 1.0594252661569492, 1.2658121316937005, 1.4937628940125933, 0.39043238917402051, 0.49222068984548584, 0.60811648588336797, 0.7385866765422433, 0.8840641933601644, 1.0449527228922362, 0.28950334532469602, 0.36673311911773132, 0.45488021504930615, 0.55431262938885573, 0.6653642278483789, 0.78834363676500241, 0.22661952350942449, 0.28843443700068938, 0.35916575964517833, 0.43910806699061661, 0.52853103472944241, 0.62768076866581679, 0.184677601592135, 0.23610461900819166, 0.29508421433871257, 0.36185689380081015, 0.4366456989944002, 0.5196643843086165])

# print tau_shape.reshape(6,6)
# print tau_scale.reshape(6,6)

n_cracks = []

for i,m in enumerate([6.0, 7.0, 8.0, 9.0, 10.0, 11.0]):
    for j,s in enumerate([0.0070, 0.0075, 0.0080, 0.0085, 0.0090, 0.0095]):
        
        scale = tau_scale[i*6+j]
        shape = tau_shape[i*6+j]

        reinf = ContinuousFibers(r=3.5e-3,
                                  tau=RV('gamma', loc=0., scale=scale, shape=shape),
                                  V_f=0.015,
                                  E_f=180e3,
                                  xi=fibers_MC(m=m, sV0=s),
                                  label='carbon',
                                  n_int=500)
        
        cb =  RandomBondCB(E_m=25e3,
                           reinforcement_lst=[reinf],
                           n_BC = 12,
                           L_max = 400)
        ctt = CompositeTensileTest(n_x = 1000,
                           L = 400,
                           cb=cb,
                           sig_mu_x= random_field.random_field)
        ctt.get_cracking_history()
        print m, s
        print 'number of cracks:', len(ctt.y)
        n_cracks.append(len(ctt.y))

print n_cracks
        



    




