import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import optimize
import os
#x = "/home/aborba/github/mb2021/Data/co2_dataset.txt"
os.getcwd()
os.chdir("/home/aborba/github/mb2021/Data/")
f = open("flevoland_1.txt","r" )
os.getcwd()
os.chdir("/home/aborba/github/mb2021/Code_python/")
# Make manipulation of the files
dados = np.loadtxt(f)
def loglike(x, z, j):
    if x[0] > 100:
        x[0] = 100
    L  = np.abs(x[0])
    mu = np.abs(x[1])
    aux1 = L * np.log(L)
    aux2 = L * sum(np.log(z[0: j])) / j
    aux3 = L * np.log(mu)
    aux4 = np.log(math.gamma(L))
    aux5 = (L / mu) * sum(z[0: j]) / j
    ll   = -(aux1 + aux2 - aux3 - aux4 - aux5)
    #ll   = (x[0] - 1)**2 + (x[1] - 3)**2
    return ll
#
def loglikd(x, z, j, n):
    if x[0] > 100:
        x[0] = 100
    L = np.abs(x[0])
    mu = np.abs(x[1])
    aux1 = L * np.log(L)
    aux2 = (L - 1) * sum(np.log(z[j: n])) / (n - j)
    aux3 = L * np.log(mu)
    aux4 = np.log(math.gamma(L))
    aux5 = (L / mu) * sum(z[j: n]) / (n - j)
    #### Beware! The signal is negative because BFGS finds the point of minimum
    ll =  -(aux1 + aux2 - aux3 - aux4 - aux5)
    return ll
#
def loglik_gauss(theta, x, N):
    mu = theta[0]
    sigma = theta[1]
    l = -(-N * np.log(np.sqrt(2*np.pi)) - N*np.log(sigma) - 0.5*np.sum((x - mu)**2/sigma**2))
    return l
#
def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)
#
k = 9
z = dados[k, :]
N = len(z)
zaux = np.zeros(N)
conta = 0
for i in range(0, N):
    if z[i] > 0:
        zaux[conta] = z[i]
        conta = conta + 1
        #
indx  = which(zaux != 0)
N = int(np.max(indx)) + 1
z =  zaux[0:N]
j = 20
varx = np.zeros(2)
intvarx = np.zeros(2)
intvarx[0] = 1
intvarx[1] = sum(z[0: j]) / j
varx[0] = 4
varx[1] = sum(z[0: j]) / j
#
#j = j - 1
print(j)
res1 = minimize(lambda varx:loglike(varx, z, j), varx, method="BFGS", \
      tol = 1e-08, \
      options={'gtol': 1e-08, \
               'eps': 1.4901161193847656e-08,\
               'maxiter': 200, \
               'disp': False,   \
               'return_all': True})
#varx[0] = 4
#varx[1] = sum(z[j: N]) / (N - j)
#res2 = minimize(lambda varx:loglikd(varx, z, j, N), varx, method="BFGS", \
#      tol = 1e-08, \
#      options={'gtol': 1e-08, \
#               'eps': 1.4901161193847656e-08,\
#               'maxiter': 200, \
#               'disp': False,   \
#               'return_all': True })
#res1 = optimize.fmin_bfgs(lambda varx:loglike(varx, z, j), varx, \
#                          norm)

#for j in range(1, N):
#    r1 = 1
#    r2 = sum(z[0: j]) / j
#    print(r2)
#    varx = np.zeros(2)
#    varx[0] = r1
#    varx[1] = r2
#    #res = minimize(lambda varx:loglike(varx, z, j), varx, method="BFGS")
#    res = minimize(lambda varx:loglik_gauss(varx, z, j), varx, method="BFGS")
    #print(rex.x)
#
#var = np.zeros(2)
#N = dados.shape[0]
#
#
#var[0] = 30
#var[1] = 10
#res = minimize(lambda var:loglik(var, x, N), var, method='BFGS')
#for canal in range(0,1):
#    print(canal)
#    for k in range(NUM_RAIOS):
    #for k in range(9, 10):
#        print(k)
#        N = RAIO
#        z = MY[k, :, canal]
#        zaux = np.zeros(N)
#        conta = 0
#        for i in range(0, N):
#            if z[i] > 0:
#                zaux[conta] = z[i]
#                conta = conta + 1
#            #
#        indx  = which(zaux != 0)
#        N = int(np.max(indx))
#        z =  zaux[1:N]
#        matdf1 =  np.zeros((N, 2))
#        matdf2 =  np.zeros((N, 2))
#        for j in range(1, N):
#            #print(j)
#        #for j in range(10, 11):
##            Le = 4.0
#            lc1 = sum(z[0: j]) / j
#            lc2 = sum(np.log(z[0: j])) / j
#            varx[0] = 4
##            res = minimize(lambda varx:loglike(varx, z, j), varx, method='BFGS')
##            matdf1[j, 0] = res.x[0]
#            matdf1[j, 1] = res.x[1]
#            mud = sum(z[j: (N + 1)]) / (N - j)
#            Ld = 4.0
#            lc1 = mud
#            lc2 = sum(np.log(z[j: (N + 1)])) / (N - j)
#            varx[0] = 4
#            varx[1] = lc1
#            res = minimize(lambda varx:loglikd(varx, z, j, N), varx, method='BFGS')
#            #res = minimize(lambda varx:loglikd(varx, z, j, N), varx, method='BFGS')
#            matdf2[j, 0] = res.x[0]
#            matdf2[j, 1] = res.x[1]
#
