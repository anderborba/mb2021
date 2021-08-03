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
### Import mod
import polsar_basics as pb
import polsar_loglikelihood as plk
import polsar_fusion as pf
import polsar_total_loglikelihood as ptl
import polsar_evidence_lib as pel

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
#
#def add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC):
#    IM  = np.zeros([nrows, ncols, ncanal])
#    for canal in range(ncanal):
#        for k in range(NUM_RAIOS):
#            ik = np.int(evidencias[k, canal])
#            ia = np.int(MXC[k, ik])
#            ja = np.int(MYC[k, ik])
#            IM[ja, ia, canal] = 1
#    return IM
## Shows the evidence
#def show_evidence(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
#	PIA=pauli.copy()
#	plt.figure(figsize=(20*img_rt, 20))
#	for k in range(NUM_RAIOS):
#    		ik = np.int(evidence[k, banda])
#    		ia = np.int(MXC[k, ik])
#    		ja = np.int(MYC[k, ik])
#    		plt.plot(ia, ja, marker='o', color="darkorange")
#	plt.imshow(PIA)
#	plt.show()

# In[20]:


## Put evidences into an image
#def add_evidence(nrows, ncols, ncanal, evidencias):
#    IM  = np.zeros([nrows, ncols, ncanal])
#    for canal in range(ncanal):
#        for k in range(NUM_RAIOS):
#            ik = np.int(evidencias[k, canal])
#            ia = np.int(MXC[k, ik])
#            ja = np.int(MYC[k, ik])
#            IM[ja, ia, canal] = 1
#    return IM



# ### A célula abaixo funciona como um main do código de fusão de evidências de borda em imagens POLSAR - ainda deverá ser editado para uma melhor compreensão do código ###
#
## Define the image and the data from the ROI in the image
imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, gt_coords = select_data()
#
## Reads the image and return the image, its shape and the number of channels
img, nrows, ncols, nc = pb.le_imagem(imagem)
#
## Plot parameter
img_rt = nrows/ncols
#
## Uses the Pauli decomposition to generate a visible image
PI = pb.show_Pauli(img, 1, 0)
#
## Define the radius in the ROI
x0, y0, xr, yr = pb.define_radiais(RAIO, NUM_RAIOS, dx, dy, nrows, ncols, alpha_i, alpha_f)

MXC, MYC, MY, IT, PI = pb.desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr)

## Define the number of channels to be used to find evidence
## and realize the fusion in the ROI
ncanal = 3
evidencias = np.zeros((NUM_RAIOS, ncanal))
## Find the evidences
## Define the number of the intensities channels
#evidencias[:, 0 : ncanal] = pel.find_evidence(RAIO, NUM_RAIOS, ncanal, MY)
evidencias[:, 0 : ncanal] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, ncanal, MY)
print(evidencias[:, 0 : ncanal])
#evidencias[:, 0 : intensities_canal] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, intensities_canal, MY)
## Put the evidences in an image
#IM = pel.add_evidence(nrows, ncols, ncanal, evidencias)
IM = pel.add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC)

## Computes fusion using mean - metodo = 1
MEDIA = pf.fusao(IM, 1, NUM_RAIOS)

## Computes fusion using pca - metodo = 2
PCA = pf.fusao(IM, 2, NUM_RAIOS)

## Computes fusion using ROC - metodo = 3
ROC = pf.fusao(IM, 3, NUM_RAIOS)

## Testing fusion using SVD - metodo = 4
#FI = pf.fusao(IM, 4, NUM_RAIOS)
SVD = pf.fusao(IM, 4, NUM_RAIOS)
#SVD=FI

## Testing fusion using SWT - metodo = 5
#FI = pf.fusao(IM, 5, NUM_RAIOS)
SWT = pf.fusao(IM, 5, NUM_RAIOS)
#SWT=FI

## Testing fusion using DWT - metodo = 6
#FI = pf.fusao(IM, 6, NUM_RAIOS)
DWT = pf.fusao(IM, 6, NUM_RAIOS)
#DWT=FI
pel.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 0)
pel.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 1)
pel.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 2)
pf.show_fusion_evidence(PI, nrows, ncols, MEDIA, img_rt)
pf.show_fusion_evidence(PI, nrows, ncols, PCA, img_rt)
pf.show_fusion_evidence(PI, nrows, ncols, ROC, img_rt)
pf.show_fusion_evidence(PI, nrows, ncols, DWT, img_rt)
pf.show_fusion_evidence(PI, nrows, ncols, SWT, img_rt)
pf.show_fusion_evidence(PI, nrows, ncols, SVD, img_rt)


## Define a variable to store the ground truth lines
#GT = np.zeros([nrows, ncols])

## The lines vector defines the lines in the ROI where ground truth information will be generated. The lines are defined
## as follows:
## The first value corresponds to the line connecting the top left corner to the top right corner. If this value is 1
## the ground truth data is computed. The second value corresponds to the next line in the quadrilateral in clockwise
## order

## Flevoland
#lines=[1,1,1,1]
## San Francisco Bay
##lines=[1,0,0,1]

## Find the ground truth data based on the Bresenham algorithm - This needs a review.
#gt_lines = pb.get_gt_lines(gt_coords, lines)

## Finds the extrem points of each line - this is done just to plot the lines using matplotlib

#gt_lines_coords=[]
#i=0
#for l in range(len(lines)):
#    if lines[l]==1:
#        x=gt_lines[i][0][0]
#        y=gt_lines[i][1][0]
#        i+=1
#        gt_lines_coords.append([x,y])

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
#i=0
#for l in range(len(lines)):
#    if lines[l]==1:
#        x0=gt_lines_coords[i][0]
#        y0=gt_lines_coords[i][1]
#        x1=gt_lines[i][0][len(gt_lines[i][0])-1]
#        y1=gt_lines[i][1][len(gt_lines[i][1])-1]
#        i=i+1
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
