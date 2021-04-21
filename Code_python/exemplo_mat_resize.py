import numpy as np
m = 16
n = 16
MAT = np.zeros((m, n))
for j in range(n):
    for i in range(m):
        MAT[i, j] = i + j
#
mm = int(m/2)
nn = int(n/2)
A = np.zeros((4, mm * nn))
AUX = np.zeros(4)
for j in range(nn):
    for i in range(mm):
        for l in range(2):
            for k in range(2):
                A[k + l * 2, i + j * mm] = MAT[i * 2 + k, j * 2 + l]
#
U, S, V = np.linalg.svd(A, full_matrices=True)
UT =  U.transpose()
T = UT @ A
TLL = np.zeros((mm, nn))
TLH = np.zeros((mm, nn))
THL = np.zeros((mm, nn))
THH = np.zeros((mm, nn))
for j in range(nn):
    for i in range(mm):
        TLL[i, j] = T[0, i + j * mm]
        TLH[i, j] = T[1, i + j * mm]
        THL[i, j] = T[2, i + j * mm]
        THH[i, j] = T[3, i + j * mm]
#
Y = []
Y.append([TLL, TLH, THL, THH])
