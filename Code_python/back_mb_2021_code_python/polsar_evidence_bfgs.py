## This function computes the indexes from a list where the condition is true
## call: get_indexes(condicao) - example: get_indexes(x>0)
import numpy as np
## Used to find border evidences
import math
from scipy.optimize import dual_annealing
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_indexes(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'get_indexes' method can only be applied to iterables.{}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)
    
    
# Total Log-likelihood function applies to the sample from l index 
# until N (Sample end). 
# Ref: Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images
# IEEE Geoscience and Remote Sensing Letters
# DOI: 10.1109/LGRS.2020.3022511
#
# Total Log-likelihood function is used to detect edge evidence.
# input: j - Reference pixel.
#        n - Sample length.
#        z - Sample.
#        matdf1 - Parameters (L, mu) until j.
#        matdf2 - Parameters (L, mu) from j until n.
# output: Total Log-likelihood function value
#
def func_obj_l_L_mu(j, z, n, matdf1, matdf2):
    j = int(np.round(j))
    mue = matdf1[j, 0]
    Le  = matdf1[j, 1]
    mud = matdf2[j, 0]
    Ld  = matdf2[j, 1]
    somaze = sum(z[0: j]) / j
    somalogze = sum(np.log(z[0: j])) / j
    somazd = sum(z[j: n]) / (n - j)
    somalogzd = sum(np.log(z[j: n])) / (n - j)
    #
    aux1 = Le * np.log(Le)
    aux2 = Le * somalogze
    aux3 = Le * np.log(mue)
    aux4 = np.log(math.gamma(Le))
    aux5 = (Le / mue) *  somaze
    #
    aux6  = Ld * np.log(Ld)
    aux7  = Ld * somalogzd
    aux8  = Ld * np.log(mud)
    aux9  = np.log(math.gamma(Ld))
    aux10 = (Ld / mud) * somazd
    a1 =  aux1 + aux2 - aux3 - aux4 - aux5
    a2 =  aux6 + aux7 - aux8 - aux9 - aux10
    #
    func_obj_l_L_mu = (j * a1 + (n - j) * a2)
    return func_obj_l_L_mu
    
    
# Log likelihood function to gamma distribution until l index.
# Ref: Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images
# IEEE Geoscience and Remote Sensing Letters
# DOI: 10.1109/LGRS.2020.3022511
#
# Log-likelihood function is used to estimate parameters (L, mu).
# input: Vector with (L, mu) to evaluate.
#        j - Reference pixel.
#        z - Sample.
# output: Log-likelihood function value
#
def loglike(x, z, j):
    L  = x[0]
    mu = x[1]
    aux1 = L * np.log(L)
    aux2 = L * sum(np.log(z[0: j])) / j
    aux3 = L * np.log(mu)
    aux4 = np.log(math.gamma(L))
    aux5 = (L / mu) * sum(z[0: j]) / j
    #### Beware! The signal is negative because BFGS routine finds the point of minimum
    ll   = -(aux1 + aux2 - aux3 - aux4 - aux5)
    return ll
    
    
# Log-likelihood gamma distribution function applies to the sample from l index 
# until N (Sample end). 
# Ref: Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images
# IEEE Geoscience and Remote Sensing Letters
# DOI: 10.1109/LGRS.2020.3022511
#
#
# Log-likelihood function is used to estimate parameters (L, mu).
# input: Vector with (L, mu) to evaluate.
#        j - Reference pixel.
#        z - Sample.
#        n - Sample size
# output: Log-likelihood function value
#
def loglikd(x, z, j, n):
    L  = x[0]
    mu = x[1]
    aux1 = L * np.log(L)
    aux2 = L * sum(np.log(z[j: n])) / (n - j)
    aux3 = L * np.log(mu)
    aux4 = np.log(math.gamma(L))
    aux5 = (L / mu) * sum(z[j: n]) / (n - j)
    #### Beware! The signal is negative because BFGS routine finds the point of minimum
    ll =  -(aux1 + aux2 - aux3 - aux4 - aux5)
    return ll
    
    
##Finds border evidences using BFGS to estimate the parameters. 
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method L-BFGS-B to estimate the gamma pdf  parameters. 
##    3) Optimization method Simulated annealing to detect edge border evidences. 
#
def find_evidence(RAIO, NUM_RAIOS, ncanal, MY):
    print("Computing evidence with bfgs - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros((NUM_RAIOS, ncanal))
    # Put limit lower bound (lb) to variables
    # Put limit upper bound (ub) to variables
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    for canal in range(ncanal):
        ## print(canal)
        for k in range(NUM_RAIOS):
            z = MY[k, :, canal]
            zaux = np.zeros(RAIO)
            conta = 0
            for i in range(RAIO):
                if z[i] > 0:
                    zaux[conta] = z[i]
                    conta = conta + 1
            #
            indx  = get_indexes(zaux != 0)
            N = int(np.max(indx)) + 1
            z =  zaux[0: N]
            matdf1 =  np.zeros((N - 1, 2))
            matdf2 =  np.zeros((N - 1, 2))
            varx = np.zeros(2)
            for j in range(1, N - 1):
                varx[0] = 1
                varx[1] = sum(z[0: j]) / j
                res = minimize(lambda varx:loglike(varx, z, j),
                                         varx,
                                         method='L-BFGS-B',
                                         bounds= bnds)
                matdf1[j, 0] = res.x[0]
                matdf1[j, 1] = res.x[1]
                #
                varx[0] = 1
                varx[1] = sum(z[j: N]) / (N - j)
                res = minimize(lambda varx:loglikd(varx, z, j, N),
                                         varx,
                                         method='L-BFGS-B',
                                         bounds= bnds)
                matdf2[j, 0] = res.x[0]
                matdf2[j, 1] = res.x[1]
            #
            #
            lw = [14]
            up = [N - 14]
            ### Beware! 
            ### The signal is negative in loglike and loglikd
            ### because BFGS routine finds the point of minimum,
            ### this fact  has like consequence changing the signal 
            ##  of the func_obj_l_L_mu function  
            ret = dual_annealing(lambda x:func_obj_l_L_mu(x,z, N, matdf1, matdf2), bounds=list(zip(lw, up)), seed=1234)
            evidencias[k, canal] = np.round(ret.x)
    return evidencias
    
    
    
## Put evidences into an image

def add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(NUM_RAIOS):
            ik = np.int(evidencias[k, canal])
            ia = np.int(MXC[k, ik])
            ja = np.int(MYC[k, ik])
            IM[ja, ia, canal] = 1
    return IM