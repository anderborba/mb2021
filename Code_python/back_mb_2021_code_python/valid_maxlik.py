import numpy as np
from scipy.optimize import minimize
import os
#x = "/home/aborba/github/mb2021/Data/co2_dataset.txt"
os.getcwd()
os.chdir("/home/aborba/github/mb2021/Data/")
f = open("co2_dataset.txt","r" )
fz = open("flevoland_1.txt","r" )
os.getcwd()
os.chdir("/home/aborba/github/mb2021/Code_python/")
# Make manipulation of the files
def loglik(theta, x, N):
    mu = theta[0]
    sigma = theta[1]
    l = -(-N * np.log(np.sqrt(2*np.pi)) - N*np.log(sigma) - 0.5*np.sum((x - mu)**2/sigma**2))
    return l
#
def loglike(x, z, j):
    #
    print(j)
    L = x[0]
    mu = x[1]
    print(x)
    aux1 = L * np.log(L)
    aux2 = (L - 1) * sum(np.log(z[0: j])) / j
    aux3 = L * np.log(mu)
    aux4 = np.log(math.gamma(L))
    aux5 = (L / mu) * sum(z[0: j]) / j
    ll   = -(aux1 + aux2 - aux3 - aux4 - aux5)
    #ll   = (x[0]- 3)**2 + (x[1] - 2)**2
    return ll
#
dados = np.loadtxt(f)
x = dados[:, 1]
dadosz = np.loadtxt(fz)
x = dados[:, 1]
var = np.zeros(2)
N = dados.shape[0]
var[0] = 1
var[1] = 1
#res = minimize(lambda var:loglik(var, x, N), var, method='BFGS')
res = minimize(lambda var:loglike(var, x, 1), var, method='BFGS')
