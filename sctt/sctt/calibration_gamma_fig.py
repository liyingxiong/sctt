from matplotlib import pyplot as plt
import numpy as np

sv0 = np.array([0.0075, 0.0080, 0.0085, 0.0090, 0.0095])
#strength
m7vf1s = np.array([15.92, 16.91, 17.67, 18.16, 18.59])
m7vf15s = np.array([23.67, 25.10, 26.38, 27.02, 27.67])
m8vf1s = np.array([15.28, 15.81, 17.13, 17.90, 18.46])
m8vf15s = np.array([23.14, 24.13, 25.67, 26.87, 27.61])
m9vf1s = np.array([14.69, 15.41, 16.24, 17.67, 18.16])
m9vf15s = np.array([22.60, 23.81, 25.02, 26.51, 27.34])
#density
m7vf1d = np.array([0.048, 0.058, 0.060, 0.065, 0.070])
m7vf15d = np.array([0.065, 0.073, 0.078, 0.090, 0.095])
m8vf1d = np.array([0.048, 0.050, 0.050, 0.055, 0.060])
m8vf15d = np.array([0.060, 0.068, 0.075, 0.078, 0.080])
m9vf1d = np.array([0.043, 0.045, 0.050, 0.050, 0.053])
m9vf15d = np.array([0.058, 0.065, 0.068, 0.070, 0.078])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')    

plt.figure()
plt.plot(sv0, m7vf1s, marker='o', label='$m=7, V_f=0.01$')
plt.plot(sv0, m7vf15s, marker='v', label='$m=7, V_f=0.015$')
plt.plot(sv0, m8vf1s, marker='<', label='$m=8, V_f=0.01$')
plt.plot(sv0, m8vf15s, marker='>', label='$m=8, V_f=0.015$')
plt.plot(sv0, m9vf1s, marker='1', label='$m=9, V_f=0.01$')
plt.plot(sv0, m9vf15s, marker='2', label='$m=9, V_f=0.015$')
plt.ylim((14., 33.))
plt.legend(loc=2)
plt.ylabel('strength [Mpa]')
plt.xlabel('$s_{V_0}$')
plt.xlim((0.0070, 0.0098))

plt.figure()
plt.plot(sv0, m7vf1d, '--', marker='.', label='$m=7, V_f=0.01$')
plt.plot(sv0, m7vf15d, '--', marker='^', label='$m=7, V_f=0.015$')
plt.plot(sv0, m8vf1d, '--', marker='3', label='$m=8, V_f=0.01$')
plt.plot(sv0, m8vf15d, '--', marker='4', label='$m=8, V_f=0.015$')
plt.plot(sv0, m9vf1d, '--', marker='s', label='$m=9, V_f=0.01$')
plt.plot(sv0, m9vf15d, '--', marker='p', label='$m=9, V_f=0.015$')
plt.ylim((0.035, 0.11))
plt.legend(loc=2)
plt.ylabel('crack density [1/mm]')
plt.xlim((0.0070, 0.0098))
plt.xlabel('$s_{V_0}$')


plt.show()



