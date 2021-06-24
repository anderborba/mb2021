import numpy as np
#
def mr_svd(M, m, n):
    # Direct SVD decomposition
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
#
def mr_isvd(Y, U):
    # Inverse SVD decomposition
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
#
def fus_svd(IM, m, n, nc):
    # Realize the SVD FUSION
    XC = []
    UC = []
    # Calculate the SVD methods for each image (channel)
    # Storage into two list
    for c in range(nc):
        X, U = mr_svd(IM[:, :, c], m, n)
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
#
m = 1000
n = 1000
nc = 3
MAT = np.zeros((m, n))
for j in range(n):
    for i in range(m):
        #MAT[i, j] = 1
        MAT[i, j] = (i + 1) + (j + 1)
        #MAT[i, j] = 1 / ((i + 1) + (j + 1))
#
IMTES = np.zeros((m, n, nc))
IMTES[:,:, 0] = MAT
IMTES[:,:, 1] = MAT
IMTES[:,:, 2] = MAT
TESTE = fus_svd(IMTES, m, n, nc)
erro = np.max(np.abs(MAT - TESTE))
#TESTE, U = mr_svd(MAT, m, n)
#TESTE    = mr_isvd(TESTE, U)
