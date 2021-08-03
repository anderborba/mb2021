from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import os
from scipy.optimize import minimize
# Function Definition
def func_pdf_gamma(x, sigma1, sigma2):
            aux1 = x[0] * np.log(x[0] / x[1])
            aux2 = (x[0] - 1) * sigma1
            aux3 =  np.log(math.gamma(x[0]))
            aux4 = (x[0] / x[1]) * sigma2
            z =  - (aux1 + aux2 - aux3 - aux4)
            return z
# Make a read
os.chdir("/home/aborba/ufal_mack/Code/Code_igarss_2021/Data")
f = open("Phantom_gamf_0.000_1_2_3.txt","r")
os.chdir("/home/aborba/bm2021/Code_python")
# Make manipulation of the files
mat = np.loadtxt(f)
#
N  = mat.shape[0]         # Line point numbers
NR = mat.shape[1]         # Radial numbers
evidencias = np.zeros(NR)
evidencias_valores = np.zeros(NR)
xev = np.arange(0, NR, 1)
#for k in range(0, NR):
for k in range(149, 150):
    print("k",k)
    z  = mat[k, 0: NR]
    matdf1 = np.zeros((N, 2))
    matdf2 = np.zeros((N, 2))
    for j in range(0, 1):
        print(j)
        sigma1 = sum(np.log(z[0: j + 1])) / (j + 1)
        sigma2 = sum(z[0: j + 1]) / (j + 1)
        r1 = 1
        r2 = sigma2
        x0 = np.array([r1, r2])
        res = minimize(func_pdf_gamma, x0, args=(sigma1, sigma2), method='BFGS')
        matdf1[j, :] = res.x
