#
import numpy as np
#
def mr_svd(MAT, m, n):
    m = int(m/2)
    n = int(n/2)
    A = np.zeros((4, m * n))
    AUX = np.zeros(4)
    for j in range(n):
        for i in range(m):
            for l in range(2):
                for k in range(2):
                    A[k + l * 2, i + j * m] = MAT[i * 2 + k, j * 2 + l]
    #
    U, S, V = np.linalg.svd(A, full_matrices=True)
    UT =  U.transpose()
    T = UT @ A
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
    Y = []
    Y.append([TLL, TLH, THL, THH])
    return Y, U
#
def mr_isvd(Y, U):
    dim = Y[0][0].shape
    mn = dim[0] * dim[1]
    T = np.zeros((4, mn))
    for j in range(nn):
        for i in range(mm):
            T[0, i + j * mm] = Y[0][0][i][j]
            T[1, i + j * mm] = Y[0][1][i][j]
            T[2, i + j * mm] = Y[0][2][i][j]
            T[3, i + j * mm] = Y[0][3][i][j]
    #
    A = U @ T
    for j in range(nn):
        for i in range(mm):
            for l in range(2):
                for k in range(2):
                    MAT[i * 2 + k, j * 2 + l] = A[k + l * 2, i + j * mm]
    return MAT
#
#def fus_coef_svd()
#    XC = []
#    UC = []
#    for c in range(nc):
#        X, U = mr_svd(IM[:, :, c], m, n)
#        XC.append(X)
#        UC.append(U)
#    XLL = sum(X1[0][0][:]) / nc
#    return
def fus_svd(IM, m, n, nc):
    for c in range(nc):
        X, U = mr_svd(IM[:, :, c], m, n)
        XC.append(X)
        UC.append(U)

        #X1, U1 = mr_svd(IM[:, :, 0], m, n)
        #X2, U2 = mr_svd(IM[:, :, 1], m, n)
        #X3, U3 = mr_svd(IM[:, :, 2], m, n)
    # Fusion
    XLL = sum(X1[0][0][:]) / nc
    XLL = (X1[0][0] + X2[0][0] + X3[0][0]) / nc
    #
    #
    # Matriz boloeana (T, F)
    #D = np.maximum(X1[0][1], X2[0][1])>= 0
    # Negacao da matriz boloeana (T, F)
    #DN = ~D
    # Matrizes Booleanas para real
    #DB = D.astype(float)
    #DBN =DN.astype(float)
    # Multiplicação elemento elemento
    #XA = DB * X1[0][1] + DBN * X2[0][1]
    #D = np.maximum(XA[0][1], X3[0][1])>= 0
    #DN = ~D
    #DB = D.astype(float)
    #DBN =DN.astype(float)
    # Multiplicação elemento elemento
    #XLH = DB * XA[0][1] + DBN * X3[0][1]
    #
    #
    # Matriz boloeana (T, F)
    #D = np.maximum(X1[0][2], X2[0][2])>= 0
    # Negacao da matriz boloeana (T, F)
    #DN = ~D
    # Matrizes Booleanas para real
    #DB = D.astype(float)
    #DBN =DN.astype(float)
    # Multiplicação elemento elemento
    #XA = DB * X1[0][2] + DBN * X2[0][2]
    #D = np.maximum(XA[0][2], X3[0][2])>= 0
    #DN = ~D
    #DB = D.astype(float)
    #DBN =DN.astype(float)
    # Multiplicação elemento elemento
    #XHL = DB * XA[0][2] + DBN * X3[0][2]
    #
    #
    # Matriz boloeana (T, F)
    #D = np.maximum(X1[0][3], X2[0][3])>= 0
    # Negacao da matriz boloeana (T, F)
    #DN = ~D
    # Matrizes Booleanas para real
    #DB = D.astype(float)
    #DBN =DN.astype(float)
    # Multiplicação elemento elemento
    #XA = DB * X1[0][3] + DBN * X2[0][3]
    #D = np.maximum(XA[0][3], X3[0][3])>= 0
    #DN = ~D
    #DB = D.astype(float)
    #DBN =DN.astype(float)
    # Multiplicação elemento elemento
    #XHH = DB * XA[0][3] + DBN * X3[0][3]
    #
    #
    #U = (U1 + U2 + U3) / nc
    #XF = []
    #XF.append([XLL, XLH, XHL, XHH])
    #IF = mr_isvd(XF, U)
    #return IF
#
m = 16
n = 16
nc = 3
MAT = np.zeros((m, n))
for j in range(n):
    for i in range(m):
        #MAT[i, j] = 1
        MAT[i, j] = i + j
#
IMTES = np.zeros((m, n, nc))
IMTES[:,:, 0] = MAT
IMTES[:,:, 1] = MAT
IMTES[:,:, 2] = MAT
TESTE = fus_svd(IMTES, m, n, nc)
#TESTE, U = mr_svd(MAT, m, n)
#TESTE    = mr_isvd(TESTE, U)
