#!/usr/bin/env python
# coding: utf-8
### OBS: Mesma versão do CodigoAndersonCompleto.ipynb
### Versao dia 23/06/2021
# ## Código de fusão de evidências de borda em imagens POLSAR ##
#
# ### Bibliotecas utilizadas ###

# In[2]:
## Import all required libraries
import numpy as np
## Used to read images in the mat format
import scipy.io as sio
## Used to equalize histograms in images
from skimage import exposure
## Used to present the images
import matplotlib as mpl
import matplotlib.pyplot as plt
## Used to find border evidences
import math
from scipy.optimize import dual_annealing
## Used in the DWT and SWT fusion methods
import pywt
#### Used to find_evidence_bfgs
from scipy.optimize import minimize
## Used


# ### Funções para ler as imagens e dados das regiões de interesse ###

# In[34]:


## This function defines the source image and all the dat related to the region where we want
## to find borders
## Defines the ROI center and the ROI boundaries. The ROI is always a quadrilateral defined from the top left corner
## in a clockwise direction.

def select_data():
    print("Select the image to be processed:")
    print("1.Flevoland - area 1")
    print("2.San Francisco")
    opcao=int(input("type the option:"))
    if opcao==1:
        print("Computing Flevoland area - region 1")
        ## Flevoland image
        ### ABB computer
        imagem="/home/aborba/github/mb2021/Data/AirSAR_Flevoland_Enxuto.mat"
        ### MM Computer
        #imagem="./Data/AirSAR_Flevoland_Enxuto.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=278
        dy=64
        ## ROI coordinates
        x1 = 157;
        y1 = 284;
        x2 = 309;
        y2 = 281;
        x3 = 310;
        y3 = 327;
        x4 = 157;
        y4 = 330;
    else:
        print("Computing San Francisco Bay area - region 1")
        ## San Francisco Bay image
        ### ABB computer
        imagem="/home/aborba/github/mb2021/Data/SanFrancisco_Bay.mat"
        ### MM Computer
        #imagem="./Data/SanFrancisco_Bay.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=50
        dy=-195
        ## ROI coordinates
        x1 = 180;
        y1 = 362;
        x2 = 244;
        y2 = 354;
        x3 = 250;
        y3 = 420;
        x4 = 188;
        y4 = 427;
    ## Radius length
    RAIO=120
    ## Number of radius used to find evidence considering a whole circunference
    NUM_RAIOS=100
    ## inicial angle to start generating the radius
    alpha_i=0.0
    ## final angle to start generating the radius
    alpha_f=2*np.pi
    ## adjust the number of radius based on the angle defined above
    if (alpha_f-alpha_i)!=(2*np.pi):
        NUM_RAIOS=int(NUM_RAIOS*(alpha_f-alpha_i)/(2*np.pi))
    gt_coords=[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    return imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, gt_coords


# In[4]:


## Read an image in the mat format
def le_imagem(img_geral):
    img=sio.loadmat(img_geral)
    img_dat=img['S']
    img_dat=np.squeeze(img_dat)
    img_shp=img_dat.shape
    ## print(img_shp)
    ncols=img_shp[1]
    nrows=img_shp[0]
    nc=img_shp[len(img_shp)-1]
    return img_dat, nrows, ncols, nc


# In[5]:


## Uses the Pauli decomposition to viaulalize the POLSAR image
def show_Pauli(data, index, control):
    Ihh = np.real(data[:,:,0])
    Ihv = np.real(data[:,:,1])
    Ivv = np.real(data[:,:,2])
    Ihh=np.sqrt(np.abs(Ihh))
    Ihv=np.sqrt(np.abs(Ihv))/np.sqrt(2)
    Ivv=np.sqrt(np.abs(Ivv))
    R = np.abs(Ihh - Ivv)
    G = (2*Ihv)
    B =  np.abs(Ihh + Ivv)
    R = exposure.equalize_hist(R)
    G = exposure.equalize_hist(G)
    B = exposure.equalize_hist(B)
    II = np.dstack((R,G,B))
    HSV = mpl.colors.rgb_to_hsv(II)
    Heq = exposure.equalize_hist(HSV[:,:,2])
    HSV_mod = HSV
    HSV_mod[:,:,2] = Heq
    Pauli_Image= mpl.colors.rgb_to_hsv(HSV_mod)
    return Pauli_Image


# ### Algoritmo de Bresenham ###

# In[6]:


## The Bresenham algorithm
## Finds out in what octant the radius is located and translate it to the first octant in order to compute the pixels in the
## radius. It translates the Bresenham line back to its original octant
def bresenham(x0, y0, xf, yf):
    x=xf-x0
    y=yf-y0
    m=10000
    ## avoids division by zero
    if abs(x) > 0.01:
        m=y*1.0/x
    ## If m < 0 than the line is in the 2nd or 4th quadrant
    ## print(x,y, m)
    if m<0:
        ## If |m| <= 1 than the line is in the 4th or in the 8th octant
        if abs(m)<= 1:
            ## If x > 0 than the line is in the 8th octant
            if x>0:
                y=y*-1
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                yp=list(np.asarray(yp)*-1)
            ## otherwise the line is in the 4th octant
            else:
                x=x*-1
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp=list(np.asarray(xp)*-1)
        ## otherwise the line is in the 3rd or 7th octant
        else:
            ## If y > 0 than the line is in the 3rd octant
            if y>0:
                x=x*-1
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
                xp=list(np.asarray(xp)*-1)
            ## otherwise the line is in the 7th octant
            else:
                y=y*-1
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
                yp=list(np.asarray(yp)*-1)
    ## otherwise the line is in the 1st or 3rd quadrant
    else:
        ## If |m| <= 1 than the line is in the 1st or 5th octant
        if abs(m)<= 1:
            ## if x > 0 than the line is in the 1st octant
            if x>0:
                ##print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
            ## otherwise the line is in the 5th octant
            else:
                x=x*-1
                y=y*-1
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp=list(np.asarray(xp)*-1)
                yp=list(np.asarray(yp)*-1)
        ## otherwise the line is in the 2nd or 6th octant
        else:
            ## If y > 0 than the line is in the 2nd octant
            if y>0:
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
            ## otherwise the line is in the 6th octant
            else:
                y=y*-1
                x=x*-1
                x,y = y,x
                ## print(x,y)
                xp,yp=bresenham_FirstOctante(x,y)
                xp,yp = yp,xp
                xp=list(np.asarray(xp)*-1)
                yp=list(np.asarray(yp)*-1)
    xp= list(np.asarray(xp) + x0)
    yp= list(np.asarray(yp) + y0)
    return xp, yp


# In[7]:


## Computes the Bresenham line in the first octant. The implementation is based on the article:
## https://www.tutorialandexample.com/bresenhams-line-drawing-algorithm/

def bresenham_FirstOctante(xf, yf):
    x=int(xf)
    y=int(yf)
    xp=[]
    yp=[]
    xp.append(0)
    yp.append(0)
    x_temp=0
    y_temp=0
    pk=2*y-x
    for i in range(x-1):
        ## print(pk)
        if pk<0:
            pk=pk+2*y
            x_temp=x_temp+1
            y_temp=y_temp
        else:
            pk=pk+2*y-2*x
            x_temp=x_temp+1
            y_temp=y_temp+1
        xp.append(int(x_temp))
        yp.append(int(y_temp))
    xp.append(x)
    yp.append(y)
    return xp, yp


# ### Funções que definem as radiais na ROI ###

# In[8]:


## Define the radius
def define_radiais(r, num_r, dx, dy, nrows, ncols, start, end):
    x0 = ncols / 2 - dx
    y0 = nrows / 2 - dy
    t = np.linspace(start, end, num_r, endpoint=False)
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    xr= np.round(x)
    yr= np.round(y)
    return x0, y0, xr, yr


# In[9]:


## Check if the extreme points of each radius are inside the image or not.
def test_XY(XC, YC, j, tam_Y, tam_X):
    if XC[j]<0:
        X=0
    elif XC[j]>=tam_X:
        X=tam_X-1
    else:
        X=XC[j]
    if YC[j]<0:
        Y=0
    elif YC[j]>=tam_Y:
        Y=tam_Y-1
    else:
        Y=YC[j]
    return int(X), int(Y)


# In[10]:


## Draw the radius in the image and determine the pixels where
## the image will be sampled using the Bresenham algorithm
def desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr):
    ## Cria vetors e matrizes de apoio
    IT = np.zeros([nrows, ncols])
    const =  5 * np.max(np.max(np.max(PI)))
    MXC = np.zeros([NUM_RAIOS, RAIO])
    MYC = np.zeros([NUM_RAIOS, RAIO])
    MY  = np.zeros([NUM_RAIOS, RAIO, nc])
    for i in range(NUM_RAIOS):
        XC, YC = bresenham(x0, y0, xr[i], yr[i])
        ##print(XC[0], YC[0], XC[len(XC)-1], YC[len(YC)-1])
        for canal in range(nc):
            Iaux = img[:, :, canal]
            dim = len(XC)
            for j in range(dim-1):
                ##print(i, canal, j, dim)
                X,Y = test_XY(XC, YC, j, nrows, ncols)
                ## print(X,Y)
                MXC[i][j] = X
                MYC[i][j] = Y
                ## invertidos
                MY[i][j][canal] = Iaux[Y][X]
                IT[Y][X] = const
                PI[Y][X] = const
    return MXC, MYC, MY, IT, PI


# ### Funções que determinam os valores de Ground Truth ###

# In[11]:


## Check the order of the line coordinates in order to call the Bresenham algorithm.
## The Bresenham algorithm assumes that x0 < x1
def verifica_coords(x0, y0, x1, y1):
    flip=0
    if x0>x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        flip=1
    return x0, y0, x1, y1, flip


# In[12]:


## Determine the ground truth lines in teh image - it is always a straight line
## The lines are genrated always from the point with the smaller x coordinate to the  point with the larger x coordinate
## Consider the example:
## given the points (10, 15) and (20, 25) generates a ground truth line from (10, 15) to (20, 25)
## given the points (20, 25) and (10, 15) generates a ground truth line from (10, 15) to (20, 25)
## Lines is a list with 4 biary values that indicates what borders of the quadrilateral should be computed
## For instance, if lines[0] = 1 finds the ground truth line that connects the points x1, y1 and x2, y2,
## if lines[1] = 1 finds the ground truth line that connects the points x2, y2 and x3, y3.
## If lines[i]=0 a no ground truth line is computed.

def get_gt_lines(gt_coords, lines):
    '''
    gt_coords:  a list of points coordinates using the xi, yi order from the ROI area
    lines: a vetor indicating the ground truth lines to be computed
    '''
    gt_lines=[]
    for l in range(len(lines)):
        if lines[l]==1:
            if l<3:
                x0, y0, x1, y1, flip=verifica_coords(gt_coords[l][0], gt_coords[l][1], gt_coords[l+1][0], gt_coords[l+1][1])
            else:
                x0, y0, x1, y1, flip=verifica_coords(gt_coords[l][0], gt_coords[l][1], gt_coords[0][0], gt_coords[0][1])
            xp, yp=bresenham(x0, y0, x1, y1)
            if flip==1:
                xp.reverse()
                yp.reverse()
            gt_lines.append([xp,yp])
    return gt_lines


# ### Funções utilizadas para a determinação das evidências de borda ###

# In[13]:


## This function computes the indexes from a list where the condition is true
## call: get_indexes(condicao) - example: get_indexes(x>0)

def get_indexes(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'get_indexes' method can only be applied to iterables.{}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)


# In[14]:
# Total Log-likelihood function applies to the sample
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
#
# Total Log-likelihood function applies to the sample
# Ref:
#
#
# Total Log-likelihood function is used to detect edge evidence.
# input: j - Reference pixel.
#        n - Sample length.
#        z - Sample.
#        matdf1 - Parameters (L, mu) until j.
#        matdf2 - Parameters (L, mu) from j until n.
# output: Total Log-likelihood function value
#
def func_obj_l_intensity_ratio_tau_rho(j, z, n, matdf1, matdf2, L):
    j = int(np.round(j))
    Le = L
    Ld = L
    print("passou dentro func obj")
    print("n - j", n - j)
    taue = matdf1[j, 0]
    taud = matdf2[j, 1]
    rhoe = matdf1[j, 0]
    rhod = matdf2[j, 1]
    aux1 = Le * np.log(taue)
    print("aux1", aux1)
    aux2 = np.log(math.gamma(2 * Le))
    print("aux2", aux2)
    print("np.abs(rhoe)", np.abs(rhoe))
    aux3 = Le * np.log(1 - np.abs(rhoe)**2)
    print("aux3", aux3)
    aux4 = 2 * np.log(math.gamma(Le))
    aux5 = sum(np.log(taue + z[0 : j])) / j
    aux6 = Le * sum(np.log(z[0 : j])) / j
    aux7 = (0.5 * (2 * Le + 1)) * sum(np.log((taue + z[0: j])**2 - 4 * taue * np.abs(rhoe)**2 * z[0: j])) / j
    soma1 = aux1 + aux2 + aux3 - aux4 + aux5 + aux6 - aux7
    aux8  = Ld * np.log(taud)
    aux9  = np.log(math.gamma(2 * Ld))
    aux10 = Ld * np.log(1 - abs(rhod)**2)
    aux11 = 2 * np.log(math.gamma(Ld))
    aux12 = sum(np.log(taud + z[j : n])) / (n - j)
    aux13 = Ld * sum(np.log(z[j : n])) / (n - j)
    aux14 = (0.5 * (2 * Ld + 1)) * sum(np.log((taud + z[j: n])**2 - 4 * taud * np.abs(rhod)**2 * z[j: n])) / (n - j)
    soma2 = aux8 + aux9 + aux10 - aux11 + aux12 + aux13 - aux14
    func_obj_l = -(soma1 * j + soma2 * (n - j))
    return func_obj_l_L_mu


# In[16]:


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


# In[35]:


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
#
# Log likelihood function to intensity ratio distribution .
# Ref: Intensity and phase statistics of multilook
# polarimetric and interferometric SAR imagery
#IEEE Transactions on Geoscience and Remote Sensing
#     ( Volume: 32, Issue: 5, Sep 1994)
# DOI: 10.1109/36.312890
#
# Log-likelihood function is used to estimate parameters (tau, rho).
# input: Vector with (tau, rhi) to evaluate.
#        L = 4 fixed
#        Ni - sample start
#        Nf - end of sample.
#        z - Sample.
# output: Log-likelihood function value
#
def loglik_intensity_ratio(x, z, Ni, Nf, L):
    tau = x[0]
    rho = x[1]
    soma1 = 0
    soma2 = 0
    soma3 = 0
    print("Nf + 1", Nf + 1)
    for k in range(Ni, Nf + 1):
        soma1 = soma1 + np.log(tau + z[k])
        soma2 = soma2 + np.log(z[k])
        soma3 = soma3 + np.log((tau + z[k])**2 - 4 * tau * np.abs(rho)**2 * z[k])
    #
    print("soma1", soma1)
    aux1 = L * np.log(tau)
    aux2 = np.log(math.gamma(2 * L))
    aux3 = L * np.log(1 - np.abs(rho)**2)
    aux4 = 2 * np.log(math.gamma(L))
    aux5 = soma1 / (Nf + 1 - Ni)
    print("Nf + 1 - Ni,", Nf + 1 - Ni)
    aux6 = L * soma2 / (Nf + 1 - Ni)
    aux7 = (0.5 * (2 * L + 1)) * soma3 / (Nf + 1 - Ni)
    print("aux1", aux1)
    print("aux2", aux2)
    print("aux3", aux3)
    print("aux4", aux4)
    print("aux5", aux5)
    print("aux6", aux6)
    print("aux7", aux7)
    #### Beware! The signal is negative because BFGS routine finds the point of minimum
    ll = -(aux1 + aux2 + aux3 - aux4 + aux5 + aux6 - aux7)
    return ll
# In[36]:


## Finds border evidences

def find_evidence(RAIO, NUM_RAIOS, ncanal, MY):
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
            indx  = get_indexes(zaux != 0)
            N = int(np.max(indx))
            z =  zaux[1:N]
            matdf1 =  np.zeros((N, 2))
            matdf2 =  np.zeros((N, 2))
            for j in range(1, N):
                mue = sum(z[0: j]) / j
                matdf1[j, 0] = mue
                matdf1[j, 1] = Le
                mud = sum(z[j: (N + 1)]) / (N - j)
                matdf2[j, 0] = mud
                matdf2[j, 1] = Ld
            #
            lw = [14]
            up = [N - 14]
            #
            ret = dual_annealing(lambda x:func_obj_l_L_mu(x,z, N, matdf1, matdf2), bounds=list(zip(lw, up)), seed=1234)
            evidencias[k, canal] = np.round(ret.x)
    return evidencias


# In[48]:


##Finds border evidences using BFGS to estimate the parameters.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the gamma pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
#
def find_evidence_bfgs(RAIO, NUM_RAIOS, ncanal, MY):
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
        ret = dual_annealing(lambda x:func_obj_l_L_mu(x,z, N, matdf1, matdf2),
                              bounds=list(zip(lw, up)),
                              seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias
##Finds border evidences using BFGS to estimate the parameters in
## intensity ratio distribution.
##Using: 1) MLE - Maximum Likelihood Estimation.
##    2) Optimization method BFGS to estimate the intensity ratio pdf  parameters.
##    3) Optimization method Simulated annealing to detect edge border evidences.
##    4) Using PDF to intensity ratio
#
def find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, ncanal, MY):
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
        indx  = get_indexes(zaux != 0)
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
            res = minimize(lambda varx:loglik_intensity_ratio(varx, z, Ni, Nf, L),
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
            res = minimize(lambda varx:loglik_intensity_ratio(varx, z, Ni, Nf, L),
                                    varx,
                                    method='L-BFGS-B',
                                    bounds= bnds)
            #
            matdf2[j, 0] = res.x[0]
            matdf2[j, 1] = res.x[1]
            #
        lw = [14]
        up = [N - 14]
        print(np.max(z))
        print(np.min(z))
        print(matdf2)
        print("passou")
        ret = dual_annealing(lambda x:func_obj_l_intensity_ratio_tau_rho(j, z, N, matdf1, matdf2, L),
                                bounds=list(zip(lw, up)),
                                seed=1234)
        evidencias[k] = np.round(ret.x)
    return evidencias

# In[20]:


## Put evidences into an image
def add_evidence(nrows, ncols, ncanal, evidencias):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(NUM_RAIOS):
            ik = np.int(evidencias[k, canal])
            ia = np.int(MXC[k, ik])
            ja = np.int(MYC[k, ik])
            IM[ja, ia, canal] = 1
    return IM


# ## Fusion Methods ##

# In[21]:


## This function actually computes an OR between evidences in all channels
def media(IM, FS):
    nrows, ncols, nc = IM.shape
    for i in range(nc):
        FS=FS+IM[:,:,i]
    ##FS=FS/nc
    return FS


# In[22]:


## This function computes the fusion of edges based on the PCA technique
def pca(IM, FS):
    nrows, ncols, nc = IM.shape
    ## vectorize the data
    C=np.zeros([nrows*ncols, nc])
    for i in range(nc):
        C[:,i]=np.reshape(IM[:,:,i],[nrows*ncols])
    ## transpose the data vector
    C=np.transpose(C)
    ## Finds the covariance matrix
    COVAR=np.cov(C)
    ## extract the eigenvalues and eigenvectors
    values, vectors=np.linalg.eig(COVAR)
    ## finds the probabilities covered by the eigenvectors
    p=values[:]*1.0/np.sum(values[:])
    ## finds the fusion points based on the probability
    aux=np.zeros([nrows,ncols])
    for i in range(nc):
        aux[:,:]=IM[:,:,i]
        FS=FS+p[i]*aux
    return FS


# In[23]:


## Returns the fraction of pixels in the intersection for the value tested
def intersection(I1, I2, test):
    nrows, ncols = I1.shape
    I=0
    ## select the test used for the intersection
    ## edge vs edge
    if test==1:
        value1=1
        value2=1
    ## edge vs n edge
    if test==2:
        value1=1
        value2=0
    ## n edge vs n edge
    if test==3:
        value1=0
        value2=0
    ## n edge vs edge
    if test==4:
        value1=0
        value2=1
    ## computes the intersection
    for i in range(nrows):
        for j in range(ncols):
            if I1[i,j]== value1 and I2[i,j]==value2:
                I=I+1
    ## print(I)
    ## computes the intersection in terms of percentage
    I=I*1.0/(nrows*ncols)
    return I


# In[24]:


## computes the average over all channels and the intersection over all channels
def compute_average(I1,I2, nc, test):
    average=np.zeros([nc])
    for j in range(nc):
        soma=0
        for i in range(nc):
            temp=intersection(I1[:,:,j], I2[:,:,i], test)
            soma=soma+temp
            ## print(soma)
        average[j]=soma/nc
    return average


# In[25]:


## finds the diagnosis line and computes the distance from each point to the diagnosis line.
## returns the closest point
def findBestFusion(TP,FP, nc, p):
    A=(p-1)/p
    C=1.0
    B=-1.0
    dist=1000
    index=-1
    for i in range(nc):
        d=abs(A*FP[i]+B*TP[i]+C)/np.sqrt(A*A+B*B)
        if d<dist:
            dist=d
            index=i
    return index


# In[26]:


## Finds teh fusion over all channels using ROC combination
def roc(IM, FS, NUM_RAIOS):
    nrows, ncols, nc=IM.shape
    V=np.zeros([nrows, ncols])
    M=np.zeros([nrows,ncols, nc])
    ## computes the image will all edge evidence found over the channels
    for i in range(nc):
        V[:,:]=V[:,:]+IM[:,:,i]
    ## finds the M images
    numPointsM1=0
    numPointsM2=0
    numPointsM3=0
    for i in range(nrows):
        for j in range(ncols):
            ## edge evidence found in at least one channel
            if V[i,j]>=1:
                M[i,j,0]=1
                numPointsM1=numPointsM1+1
            ## edge evidence found in at least two channels
            if V[i,j]>=2:
                M[i,j,1]=1
                numPointsM2=numPointsM2+1
            ## edge evidence found in at least three channels
            if V[i,j]>=3:
                M[i,j,2]=1
                numPointsM3=numPointsM3+1
    ## print("# marks M1 = ", numPointsM1)
    ## print("# marks M2 = ", numPointsM2)
    ## print("# marks M3 = ", numPointsM3)
    ## finds the average of true positives
    tp=compute_average(M, IM, nc, 1)
    ## print("true positives = ", tp)
    ## finds the average of the false positives
    fp=compute_average(M, IM, nc, 2)
    ## print("false positives = ", fp)
    ## finds the average of true negatives
    tn=compute_average(M, IM, nc, 3)
    ## print("true negatives = ", tn)
    ## finds the average of false negatives
    fn=compute_average(M, IM, nc, 4)
    ## print("false negatives = ", fn)
    ## computes the average true positive rates and average false positive rates
    TP=np.zeros([nc])
    FP=np.zeros([nc])
    for i in range(nc):
        TP[i]=tp[i]/(tp[i]+fn[i])
        FP[i]=1.0-(tn[i]/(fp[i]+tn[i]))
    ## print("True Positives = ", TP)
    ## print("False Positives = ", FP)
    ## finds the value of P
    p=NUM_RAIOS*1.0/(nrows*ncols)
    ## finds the index of the best fusion image
    index=findBestFusion(TP,FP, nc, p)
    FS=M[:,:,index]
    return FS


# In[27]:


def dwt(E, m, n, nc):
    # Autors: Anderson Borba and Maurício Marengoni - Version 1.0 (04/12/2021)
    # Discrete wavelet transform Fusion
    # Input: E     - (m x n x nc) Data with one image per channel
    #        m x n - Image dimension
    #        nc    - Channels number
    # Output: F - Image fusion
    #
    # Calculates DWT to each channel nc
    # Set a list with (mat, tuple) coefficients
    cA = []
    for canal in range(nc):
        cAx, (cHx, cVx, cDx) = pywt.dwt2(E[ :, :, canal], 'db2')
        cA.append([cAx, (cHx, cVx, cDx)])
    #
    # Fusion Method
    # Calculates average to all channels with the coefficients cA from DWT transform
    cAF = 0
    for canal in range(nc):
        cAF = cAF + cA[canal][0]
    cAF = cAF / nc
    #
    # Calculates maximum to all channels with the coefficients cH, cV e Cd from DWT transform
    cHF = np.maximum(cA[0][1][0], cA[1][1][0])
    cVF = np.maximum(cA[0][1][1], cA[1][1][1])
    cDF = np.maximum(cA[0][1][2], cA[1][1][2])
    for canal in range(2, nc):
        cHF = np.maximum(cHF, cA[canal][1][0])
        cVF = np.maximum(cVF, cA[canal][1][1])
        cDF = np.maximum(cDF, cA[canal][1][2])
    #
    # Set the fusion coefficients like (mat, tuple)
    fus_coef = cAF, (cHF, cVF, cDF)
    #
    #Use the transform DWT inverse to obtain the fusion image
    F = pywt.idwt2(fus_coef, 'db2')
    return F


# In[28]:


## Implements the MR-SWT Fusion
def swt(E, m, n, nc):
# Stationary wavelet transform Fusion
# Input: E     - (m x n x nc) Data with one image per channel
#        m x n - Image dimension
#        nc    - Channels number
# Output: F - Image fusion
#
# Calculates SWT to each channel nc
# Set a list with (mat, tuple) coefficients
    cA = []
    lis = []
    for canal in range(nc):
        lis = pywt.swt2(E[ :, :, canal], 'sym2', level= 1, start_level= 0)
        cA.append(lis)
#
# Fusion Method
# Calculates the average for all channels with the coefficients cA from the SWT transform
    cAF = 0
    for canal in range(nc):
        cAF = cAF + cA[canal][0][0]
    cAF = cAF/nc
#
# Calculates the maximum for all channels with the coefficients cH, cV e Cd from the SWT transform
    cHF = np.maximum(cA[0][0][1][0], cA[0][0][1][0])
    cVF = np.maximum(cA[1][0][1][1], cA[1][0][1][1])
    cDF = np.maximum(cA[2][0][1][2], cA[2][0][1][2])
    for canal in range(2, nc):
        cHF = np.maximum(cHF, cA[canal][0][1][0])
        cVF = np.maximum(cVF, cA[canal][0][1][1])
        cDF = np.maximum(cDF, cA[canal][0][1][2])
#
# Set a list with the fusion coefficients like (mat, tuple)
    cF = []
    cF.append([cAF, (cHF, cVF, cDF)])
    F = pywt.iswt2(cF, 'sym2')
    return F


# In[29]:


def mr_svd(M, m, n):
    # Direct SVD decomposition multi-resolution
    # Input: M     - (m x n) matrix to SVD decomposition
    #        m x n - Image dimension
    # Output: Return list Y decomposition and matrix U of the SVD decomposition
    #         Where are TLL, TLH, THL, and THH into a list Y
    #
    # Set multi-resolution two level
    m = int(m/2)
    n = int(n/2)
    # Set md to two level SVD decomposition IM.LL, IM.LH, IM.HL, and IM.HH
    # Obs: Each decomposition level split the initial image into 4 matrix
    md = 4
    # Resize M into matrix A[4, m * n]
    A = np.zeros((md, m * n))
    for j in range(n):
        for i in range(m):
            for l in range(2):
                for k in range(2):
                    A[k + l * 2, i + j * m] = M[i * 2 + k, j * 2 + l]
    #
    # Calculate SVD decomposition to A
    U, S, V = np.linalg.svd(A, full_matrices=False)
    UT =  U.transpose()
    T = UT @ A
    # Set each line of T into a vector TLL, TLH, THL, and THH
    TLL = np.zeros((m, n))
    TLH = np.zeros((m, n))
    THL = np.zeros((m, n))
    THH = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            TLL[i, j] = T[0, i + j * m]
            TLH[i, j] = T[1, i + j * m]
            THL[i, j] = T[2, i + j * m]
            THH[i, j] = T[3, i + j * m]
    #
    # Put TLL, TLH, THL, and THH into a list Y
    Y = []
    Y.append(TLL)
    Y.append(TLH)
    Y.append(THL)
    Y.append(THH)
    # Return Y decomposition and matrix U of the SVD decomposition
    return Y, U


# In[30]:


def mr_isvd(Y, U):
    # Inverse SVD decomposition multi-resolution
    # Input: List Y with coeficients and matrix U fusion to SVD inverse decomposition
    #         Where TLL, TLH, THL, and THH are into a list Y
    # Output: Image fusion
    # Define dimension
    dim = Y[0].shape
    m = dim[0]
    n = dim[1]
    mn = dim[0] * dim[1]
    # Put list Y into matrix T[4, m * n]
    # Obs: Each decomposition level split the initial image into 4 matrix
    #
    T = np.zeros((4, mn))
    for j in range(n):
        for i in range(m):
            T[0, i + j * m] = Y[0][i][j]
            T[1, i + j * m] = Y[1][i][j]
            T[2, i + j * m] = Y[2][i][j]
            T[3, i + j * m] = Y[3][i][j]
    #
    # Inverse SVD
    A = U @ T
    # Put A into matrix M
    M = np.zeros((2 * m, 2 * n))
    for j in range(n):
        for i in range(m):
            for l in range(2):
                for k in range(2):
                    M[i * 2 + k, j * 2 + l] = A[k + l * 2, i + j * m]
    # Return the image M
    return M


# In[31]:


def svd(E, m, n, nc):
    # SVD multi-resolution Fusion
    # Input: E     - (m x n x nc) Data with one image per channel
    # Output: FS - Image fusion
    # Computes the SVD FUSION
    XC = []
    UC = []
    # Calculate the SVD methods for each image (channel)
    # Storage into two list
    for c in range(nc):
        X, U = mr_svd(E[:, :, c], m, n)
        XC.append(X)
        UC.append(U)
    #
    # Set de dimension
    mr = int(m / 2)
    nr = int(n / 2)
    SOMA = np.zeros((mr, nr))
    XLL  = np.zeros((mr, nr))
    # Calculate the average in alls decompositions X.LL (among channel)
    for c in range(nc):
        SOMA = SOMA + XC[c][0]
    XLL = SOMA / nc
    #
    XF = []
    XF.append(XLL)
    #
    # Obs: Each decomposition level split the initial image into 4 matrix
    nd = 4
    # Calculate the maximum in alls decompositions X.LH, X.HL, and X.HH (among channel)
    for c in range(1, nd):
        D = np.maximum(XC[0][c], XC[1][c])>= 0
        # Element-wise multiplication, and rule to fusion
        XA = D * XC[0][c] + ~D * XC[1][c]
        D = np.maximum(XA, XC[2][c])>= 0
        # Element-wise multiplication, and rule to fusion
        COEF = D * XA + ~D * XC[2][c]
        XF.append(COEF)
    #
    # Rule fusion to matriz list UC
    SOMA1 = np.zeros((4, 4))
    UF    = np.zeros((4, 4))
    for c in range(nc):
        SOMA1 = SOMA1 + UC[c]
    UF = SOMA1 / nc
    IF = mr_isvd(XF, UF)
    return IF


# In[32]:


def fusao(IM, metodo, NUM_RAIOS):
    nrows, ncols, nc = IM.shape
    FS=np.zeros([nrows,ncols])
    if metodo==1:
        print("finding fusion using Mean")
        FS=media(IM, FS)
    if metodo==2:
        print("finding fusion using PCA")
        FS=pca(IM, FS)
    if metodo==3:
        print("finding fusion using ROC")
        FS=roc(IM, FS, NUM_RAIOS)
    if metodo==4:
        print("finding fusion using SVD")
        FS=svd(IM, nrows, ncols, nc )
    if metodo==5:
        print("finding fusion using SWT")
        FS=swt(IM, nrows, ncols, nc)
    if metodo==6:
        print("finding fusion using DWT")
        FS=dwt(IM, nrows, ncols, nc)
    if metodo==7:
        print("finding fusion using Majority")
        FS=majority(IM, FS)
    return FS


# ## Metrics ##

# ### A célula abaixo funciona como um main do código de fusão de evidências de borda em imagens POLSAR - ainda deverá ser editado para uma melhor compreensão do código ###

# In[49]:


## Define the image and the data from the ROI in the image
imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, gt_coords = select_data()
## Reads the image and return the image, its shape and the number of channels
img, nrows, ncols, nc = le_imagem(imagem)

##print(ncols, nrows, nc)
## Plot parameter
kdw = nrows/ncols

## Uses the Pauli decomposition to generate a visible image
PI=show_Pauli(img, 1, 0)

## Define the radius in the ROI
x0, y0, xr, yr=define_radiais(RAIO, NUM_RAIOS, dx, dy, nrows, ncols, alpha_i, alpha_f)

MXC, MYC, MY, IT, PI=desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr)

## Define the number of channels to be used to find evidence
## and realize the fusion in the ROI
ncanal = 5
evidencias = np.zeros((NUM_RAIOS, ncanal))
## Find the evidences
## Define the number of the intensities channels
intensities_canal = 3


# rotina teste
z = np.ones(RAIO)
k = 9
j = 19
Ni = 0
Nf = j
xx = np.zeros(2)
xx[0] = 0.5
xx[1] = 0.5
L = 4
value = loglik_intensity_ratio(xx, z, Ni, Nf, L)
print(value)
lbtau = 0.00000001
ubtau = 100
lbrho = -0.9999999
ubrho =  0.9999999
bnds = ((lbtau, ubtau), (lbrho, ubrho))
res = minimize(lambda xx:loglik_intensity_ratio(xx, z, Ni, Nf, L),
                        xx,
                        method='L-BFGS-B',
                        bounds= bnds)

#evidencias[:, 0 : intensities_canal] = find_evidence_bfgs(RAIO, NUM_RAIOS, intensities_canal, MY)
#evidencias[:, ncanal - 2] = find_evidence_bfgs_span(RAIO, NUM_RAIOS, intensities_canal , MY)
evidencias[:, ncanal - 1] = find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, intensities_canal , MY)
## Put the evidences in an image
IM=add_evidence(nrows, ncols, ncanal, evidencias)

## Computes fusion using mean - metodo = 1
#MEDIA=fusao(IM, 1, NUM_RAIOS)

## Computes fusion using pca - metodo = 2
#PCA=fusao(IM, 2, NUM_RAIOS)

## Computes fusion using ROC - metodo = 3
#ROC=fusao(IM, 3, NUM_RAIOS)

## Testing fusion using SVD - metodo = 4
#FI=fusao(IM, 4, NUM_RAIOS)
#SVD=FI

## Testing fusion using SWT - metodo = 5
#FI=fusao(IM, 5, NUM_RAIOS)
#SWT=FI

## Testing fusion using DWT - metodo = 6
#FI=fusao(IM, 6, NUM_RAIOS)
#DWT=FI

## Define a variable to store the ground truth lines
#GT = np.zeros([nrows, ncols])

## The lines vector defines the lines in the ROI where ground truth information will be generated. The lines are defined
## as follows:
## The first value corresponds to the line connecting the top left corner to the top right corner. If this value is 1
## the ground truth data is computed. The second value corresponds to the next line in the quadrilateral in clockwise
## order

## Flevoland
lines=[1,1,1,1]
## San Francisco Bay
##lines=[1,0,0,1]

## Find the ground truth data based on the Bresenham algorithm - This needs a review.
gt_lines=get_gt_lines(gt_coords, lines)

## Finds the extrem points of each line - this is done just to plot the lines using matplotlib

gt_lines_coords=[]
i=0
for l in range(len(lines)):
    if lines[l]==1:
        x=gt_lines[i][0][0]
        y=gt_lines[i][1][0]
        i+=1
        gt_lines_coords.append([x,y])

## The prints below are just a check point - the values of the first print should be equal to the second print
## The third print shows the positions of the x and y from Bresenham for the ground truth lines.
##print(gt_lines_coords)
##print(gt_coords)
##print(gt_lines)

#plt.figure(figsize=(20 * kdw,20))
## Plots the image center point
#plt.plot(ncols/2, nrows/2, marker='v', color="blue")
#plt.plot(235, 312, marker='v', color="blue")
## Plot the points of the ROI
#plt.plot(gt_coords[0][0], gt_coords[0][1], marker='o', color="red")
#plt.plot(gt_coords[1][0], gt_coords[1][1], marker='o', color="yellow")
#plt.plot(gt_coords[2][0], gt_coords[2][1], marker='o', color="black")
#plt.plot(gt_coords[3][0], gt_coords[3][1], marker='o', color="white")

## Shows the ground truth lines selected
i=0
for l in range(len(lines)):
    if lines[l]==1:
        x0=gt_lines_coords[i][0]
        y0=gt_lines_coords[i][1]
        x1=gt_lines[i][0][len(gt_lines[i][0])-1]
        y1=gt_lines[i][1][len(gt_lines[i][1])-1]
        i=i+1
#        plt.plot([x0, x1], [y0, y1], color="green")
## shows the Pauli image
#plt.imshow(PI)
#plt.show()


# In[50]:

#PIA = []
#PIA=show_Pauli(img, 1, 0)
#plt.figure(figsize=(20*kdw, 20))
#for k in range(NUM_RAIOS):
#    ik = np.int(evidencias[k, 0])
#    ia = np.int(MXC[k, ik])
#    ja = np.int(MYC[k, ik])
#    plt.plot(ia, ja, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[51]:


## Shows the evidence for the hv channel
#plt.figure(figsize=(20*kdw,20))
#for k in range(NUM_RAIOS):
#    ik = np.int(evidencias[k, 1])
#    ia = np.int(MXC[k, ik])
#    ja = np.int(MYC[k, ik])
#    plt.plot(ia, ja, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[52]:

## Shows the evidence for the vv channel
#plt.figure(figsize=(20*kdw,20))
#for k in range(NUM_RAIOS):
#    ik = np.int(evidencias[k, 2])
#    ia = np.int(MXC[k, ik])
#    ja = np.int(MYC[k, ik])
#    plt.plot(ia, ja, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()
#
## Shows the evidence for the pdf span
#plt.figure(figsize=(20*kdw,20))
#for k in range(NUM_RAIOS):
#    ik = np.int(evidencias[k, 3])
#    ia = np.int(MXC[k, ik])
#    ja = np.int(MYC[k, ik])
#    plt.plot(ia, ja, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()
#
## Shows the evidence for the intensity ratio pdf
#plt.figure(figsize=(20*kdw,20))
#for k in range(NUM_RAIOS):
#    ik = np.int(evidencias[k, 4])
#    ia = np.int(MXC[k, ik])
#    ja = np.int(MYC[k, ik])
#    plt.plot(ia, ja, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[53]:


## Shows the mean fusion image
#plt.figure(figsize=(20*kdw, 20))
#for i in range(nrows):
#    for j in range(ncols):
#        if(MEDIA[i,j] != 0):
#            plt.plot(j,i, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[55]:


## Shows the PCA fusion image
#plt.figure(figsize=(20*kdw, 20))
#for i in range(nrows):
#    for j in range(ncols):
#        if(PCA[i,j] != 0):
#            plt.plot(j,i, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[56]:


## Shows the ROC fusion image
#plt.figure(figsize=(20*kdw, 20))
#for i in range(nrows):
#    for j in range(ncols):
#        if(ROC[i,j] != 0):
#            plt.plot(j,i, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[44]:


## Shows the SVD fusion image
#plt.figure(figsize=(20*kdw, 20))
#for i in range(nrows):
#    for j in range(ncols):
#        if(SVD[i,j] != 0):
#            plt.plot(j,i, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[45]:


## Shows the SWT fusion image
#plt.figure(figsize=(20*kdw, 20))
#for i in range(nrows):
#    for j in range(ncols):
#        if(SWT[i,j] != 0):
#            plt.plot(j,i, marker='o', color="darkorange")
#plt.imshow(PIA)
#plt.show()


# In[46]:


## Shows the DWT fusion image
#plt.figure(figsize=(20*kdw, 20))
#for i in range(nrows):
#    for j in range(ncols):
#        if(DWT[i,j] != 0):
#            plt.plot(j,i, marker='o', color="darkorange")
#plt.imshow(PIA)
#print(evidencias)
#plt.show()


# In[54]:



# In[ ]:
