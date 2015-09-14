import numpy as np
from scipy.optimize import brentq
from matplotlib import pyplot as plt

Em = 10e3 #matrix modulus 
Ef = 180e3 #fiber modulus
vf = 0.01 #reinforcement ratio
T = 12. #bond intensity
sig_cu = 18. #[MPa]
x = np.linspace(0, 1000, 1000) #specimen discretization
sig_mu_x = np.linspace(3.0, 4.5, 1000) #matrix strength field

def cb(z, sig_c): #Eq.(3) and Eq. (9)
    sig_m = np.minimum(z * T * vf / (1 - vf), Em*sig_c/(vf*Ef + (1-vf)*Em)) #matrix stress
    esp_f = (sig_c-sig_m) / vf / Ef #reinforcement strain
    return  sig_m, esp_f

def get_z_x(x, XK): #Eq.(5)
    z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
    return np.amin(z_grid, axis=1)

def get_lambda_z(sig_mu, z):
    fun = lambda sig_c: sig_mu - cb(z, sig_c)[0]
    try: # search for the local crack load level 
        return brentq(fun, 0, sig_cu)
    except: # solution not found (shielded zone) return the ultimate composite stress
        return sig_cu

def get_sig_c_K(z_x):
    get_lambda_x = np.vectorize(get_lambda_z)
    lambda_x = get_lambda_x(sig_mu_x, z_x) #Eq. (6)
    y_idx = np.argmin(lambda_x) #Eq. (7) and Eq.(8)
    return lambda_x[y_idx], x[y_idx]

def get_cracking_history():
    XK = [0.] #position of the first crack
    sig_c_K = [0., 3.0]
    eps_c_K = [0., 3.0/(vf*Ef + (1-vf)*Em)]
    while True:
        z_x = get_z_x(x, XK)
        sig_c_k, y_i = get_sig_c_K(z_x)
        if sig_c_k == sig_cu: break
        XK.append(y_i)
        sig_c_K.append(sig_c_k)
        eps_c_K.append(np.trapz(cb(get_z_x(x, XK), sig_c_k)[1], x)/1000.) #Eq. (10)
    sig_c_K.append(sig_cu)
    eps_c_K.append(np.trapz(cb(get_z_x(x, XK), sig_cu)[1], x)/1000.)
    return sig_c_K, eps_c_K

sig_c_K, eps_c_K = get_cracking_history()
plt.plot(eps_c_K, sig_c_K)
# plt.plot([0.0, sig_cu/(Ef*vf)], [0.0, sig_cu])
# plt.axis('off')
plt.show()




