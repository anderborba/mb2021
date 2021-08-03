### Version 03/08/2021
# Article GRSL
# Ref:
# A. A. De Borba,
# M. Marengoni and
# A. C. Frery,
# "Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images,"
# in IEEE Geoscience and Remote Sensing Letters,
#doi: 10.1109/LGRS.2020.3022511.
# bibtex
#@ARTICLE{9203845,
#  author={De Borba, Anderson A. and Marengoni, Maurício and Frery, Alejandro C.},
#  journal={IEEE Geoscience and Remote Sensing Letters},
#  title={Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images},
#  year={2020},
#  volume={},
#  number={},
#  pages={1-5},
#  doi={10.1109/LGRS.2020.3022511}}
#
import numpy as np
## Used to present the images
import matplotlib as mpl
import matplotlib.pyplot as plt
## Used to find border evidences
import math
from scipy.optimize import dual_annealing
#### Used to find_evidence_bfgs
from scipy.optimize import minimize
#
import polsar_basics as pb
import polsar_loglikelihood as plk
import polsar_total_loglikelihood as ptl
#
## Finds border evidences
def find_evidence(RAIO, NUM_RAIOS, ncanal, MY, lim):
    print("Computing evidence - this might take a while")
    z = np.zeros(RAIO)
    Le = 4
    Ld = 4
    evidencias = np.zeros((NUM_RAIOS, ncanal))
    for canal in range(ncanal):
        for k in range(NUM_RAIOS):
            z = MY[k, :, canal]
            zaux = np.zeros(RAIO)
            conta = 0
            for i in range(RAIO):
                if z[i] > 0:
                    zaux[conta] = z[i]
                    conta = conta + 1
            #
            indx  = pb.get_indexes(zaux != 0)
            N = int(np.max(indx))
            z =  zaux[1:N]
            matdf1 =  np.zeros((N, 2))
            matdf2 =  np.zeros((N, 2))
            for j in range(1, N):
                mue = sum(z[0: j]) / j
                matdf1[j, 0] = Le
                matdf1[j, 1] = mue
                mud = sum(z[j: (N + 1)]) / (N - j)
                matdf2[j, 0] = Ld
                matdf2[j, 1] = mud
            #
            lw = [lim]
            up = [N - lim]
            #
            ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
            evidencias[k, canal] = np.round(ret.x)
    return evidencias
#
##Finds border evidences using BFGS to estimate the parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the gamma pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
#
def find_evidence_bfgs(RAIO, NUM_RAIOS, ncanal, MY, lim):
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
        #for k in range(50, 51):
        #    print(k)
            z = MY[k, :, canal]
            zaux = np.zeros(RAIO)
            conta = 0
            for i in range(RAIO):
                if z[i] > 0:
                    zaux[conta] = z[i]
                    conta = conta + 1
                #
            indx  = pb.get_indexes(zaux != 0)
            N = int(np.max(indx))
            z =  zaux[0: N]
            matdf1 =  np.zeros((N, 2))
            matdf2 =  np.zeros((N, 2))
            varx = np.zeros(2)
            for j in range(1, N):
                varx[0] = 1
                varx[1] = sum(z[0: j]) / j
                res = minimize(lambda varx:plk.loglike(varx, z, j),
                                         varx,
                                         method='L-BFGS-B',
                                         bounds= bnds)
                matdf1[j, 0] = res.x[0]
                matdf1[j, 1] = res.x[1]
                #
                varx[0] = 1
                varx[1] = sum(z[j: N]) / (N - j)
                res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                                         varx,
                                         method='L-BFGS-B',
                                         bounds= bnds)
                matdf2[j, 0] = res.x[0]
                matdf2[j, 1] = res.x[1]
                #
                #
            #print(matdf1)
            #pobj = np.zeros(RAIO)
            #for j in range(1, N - 1):
            #    pobj[j] = ptl.func_obj_l_L_mu(j,z, N, matdf1, matdf2)
            #plt.plot(pobj[1: N - 1])
            lw = [lim]
            up = [N - lim]
            ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                                 bounds=list(zip(lw, up)),
                                 seed=1234)
            evidencias[k, canal] = np.round(ret.x)
    return evidencias
#
##Finds border evidences using BFGS to estimate the parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the gamma pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to span
#
def find_evidence_bfgs_span(RAIO, NUM_RAIOS, ncanal, MY):
    print("Computing evidence with bfgs to span PDF - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros(NUM_RAIOS)
    lb = 0.00000001
    ub = 10
    bnds = ((lb, ub), (lb, ub))
    for k in range(NUM_RAIOS):
        print(k)
        zaux = np.zeros(RAIO)
        z = MY[k, :, 0] + 2 * MY[k, :, 1] + MY[k, :, 2]
        conta = 0
        for i in range(RAIO):
            if z[i] > 0:
                zaux[conta] = z[i]
                conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx)) + 1
        z =  zaux[0: N]
        matdf1 =  np.zeros((N - 1, 2))
        matdf2 =  np.zeros((N - 1, 2))
        varx = np.zeros(2)
        for j in range(1, N - 1):
            varx[0] = 1
            varx[1] = sum(z[0: j]) / j
            res = minimize(lambda varx:plk.loglike(varx, z, j),
                            varx,
                            method='L-BFGS-B',
                            bounds= bnds)
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 1
            varx[1] = sum(z[j: N]) / (N - j)
            res = minimize(lambda varx:plk.loglikd(varx, z, j, N),
                            varx,
                            method='L-BFGS-B',
                            bounds= bnds)
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            #
            #
        lw = [14]
        up = [N - 14]
        ret = dual_annealing(lambda x:ptl.func_obj_l_L_mu_bfgs(x,z, N, matdf1, matdf2),
                              bounds=list(zip(lw, up)),
                              seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias
#
##Finds border evidences using BFGS to estimate the parameters in
## intensity ratio distribution.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the intensity ratio pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to intensity ratio
#
def find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, ncanal, MY, lim):
    print("Computing evidence with bfgs to intensity ratio pdf - this might take a while")
    z = np.zeros(RAIO)
    evidencias = np.zeros((NUM_RAIOS, ncanal))
    #lb = 0.00000001
    #ub = 10
    #bnds = ((lb, ub), (lb, ub))
    lbtau = 0.00000001
    ubtau = 100
    lbrho = -0.9999999
    ubrho =  0.9999999
    bnds = ((lbtau, ubtau), (lbrho, ubrho))
    # Set L = 4 fixed
    L = 4
    for k in range(NUM_RAIOS):
        #for k in range(9, 10):
            #print(k)
            #z = MY[k, :, canal]
        zaux = np.zeros(RAIO)
        conta = 0
        for i in range(RAIO):
            if MY[k, i, 0] > 0 and MY[k, i, 1] > 0:
                    zaux[conta] = MY[k, i, 0] / MY[k, i, 1]
                    conta = conta + 1
        #
        indx  = pb.get_indexes(zaux != 0)
        N = int(np.max(indx)) + 1
        z =  zaux[0: N]
        matdf1 =  np.zeros((N, 2))
        matdf2 =  np.zeros((N, 2))
        varx = np.zeros(2)
        for j in range(1, N):
            varx[0] = 0.5
            varx[1] = 0.1
            Ni = 0
            Nf = j
            res = minimize(lambda varx:plk.loglik_intensity_ratio(varx, z, Ni, Nf, L),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf1[j, 0] = res.x[0]
            matdf1[j, 1] = res.x[1]
            #
            varx[0] = 0.5
            varx[1] = 0.1
            Ni = j
            Nf = N - 1
            res = minimize(lambda varx:plk.loglik_intensity_ratio(varx, z, Ni, Nf, L),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            #
        lw = [lim]
        up = [N - lim]
        ret = dual_annealing(lambda x:ptl.func_obj_l_intensity_ratio_tau_rho(j, z, N, matdf1, matdf2, L),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias
# Set evidence in an image
def add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(NUM_RAIOS):
            ik = np.int(evidencias[k, canal])
            ia = np.int(MXC[k, ik])
            ja = np.int(MYC[k, ik])
            IM[ja, ia, canal] = 1
    return IM
#
## Shows the evidence
def show_evidence(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
	PIA=pauli.copy()
	plt.figure(figsize=(20*img_rt, 20))
	for k in range(NUM_RAIOS):
    		ik = np.int(evidence[k, banda])
    		ia = np.int(MXC[k, ik])
    		ja = np.int(MYC[k, ik])
    		plt.plot(ia, ja, marker='o', color="darkorange")
	plt.imshow(PIA)
	plt.show()
