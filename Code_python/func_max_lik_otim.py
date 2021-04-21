import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy import optimize
import scipy.special as sp
#
def loglike_folga(x, lc1, lc2):
    high = 10e10
    low  = 0
    logx0 = np.where(x[0] > low and  x[0] < high, np.log(x[0]), 0)
    aux1 = x[0] * logx0
    aux2 = x[0] * lc2
    aux3 = x[0] * np.log(x[1])
    aux4 = np.log(math.gamma(x[0]))
    aux5 = (x[0] / x[1]) * lc1
    ll   = aux1 + aux2 - aux3 - aux4 - aux5
    #ll = (x[0] - 5)**2 + (x[1] - 6)**2
    return ll
#
def loglike(x, lc1, lc2):
    aux1 = x[0] * np.log(x[0])
    aux2 = x[0] * lc1
    aux3 = x[0] * np.log(x[1])
    aux4 = np.log(math.gamma(x[0]))
    aux5 = (x[0] / x[1]) * lc2
    ll   = aux1 + aux2 - aux3 - aux4 - aux5
    #ll = (x[0] - 5)**2 + (x[1] - 6)**2
    return ll
#
def der_loglike_tes_folga(x, lc1, lc2):
    ### der[0] derivada em relação a L
    ### der[1] derivada em relação a mu
    high = 10e10
    low  = 0
    logx0    = np.where(x[0] > low and  x[0] < high, np.log(x[0]), 0)
    logx1    = np.where(x[1] > low and  x[1] < high, np.log(x[1]), 0)
    der = np.zeros(2)
    der[0]   = logx0 + 1 - logx1 - (1 / math.gamma(x[0])) * sp.digamma(x[0]) + lc1 - lc2 / x[1]
    der[1]   = - x[0] / x[1] + (x[0] / x[1]**2) * lc2
    return der
#
def der_loglike(x, lc1, lc2):
    ### der[0] derivada em relação a L
    ### der[1] derivada em relação a mu
    der = np.zeros(2)
    der[0]   = np.log(x[0] / x[1]) + x[1] - sp.digamma(x[0]) + lc1 - (lc2 / x[1])
    der[1]   = -(x[0] / x[1]) + (x[0] / x[1]**2) * lc2
    return der
#
lc1 = -7.101201
lc2 = 0.0010654
#
c1 = 4.0
c2 = lc2
x = np.zeros(2)
x[0] = c1
x[1] = c2
#res = optimize.fmin_bfgs(lambda x:loglike(x, lc1, lc2), x, jac =lambda x:der_loglike_tes(x, lc1, lc2))
#res = minimize(lambda varx:loglike(x, lc1, lc2), x, method='BFGS', jac = lambda x:der_loglike_tes(x, lc1, lc2))
#res = minimize(lambda x:loglike(x, lc1, lc2), x, method='BFGS', jac = lambda x:der_loglike_tes(x, lc1, lc2), options={'disp': True, 'maxiter': 10})
res = minimize(lambda x:loglike(x, lc1, lc2), x, method='BFGS', options={'disp': True, 'maxiter': 1})
#res = minimize(lambda x:loglike(x, lc1, lc2), x, method='CG', options={'disp': True, 'maxiter': 10})
