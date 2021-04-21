import numpy as np
D  = np.zeros((3, 3))
X1 = np.zeros((3, 3))
X2 = np.zeros((3, 3))
X3 = np.zeros((3, 3))
X1 = np.arange(9)
X2 = np.arange(9) - 10
X3 = -np.arange(9)
X4 = 5 * np.ones((3,3))
X5 = np.ones((3,3))
X4[0,1] = -10
X4[0,2] = -10
X4[1,2] = -10
X5[1,0] = -5
X5[2,0] = -5
X5[2,1] = -5
X6 = np.ones((3,3))
X6[1,0] = -3
X6[2,0] = -10
AUX1 = np.maximum(X1, X2)
AUX2 = np.maximum(X5, X6)
W = np.maximum(X5, X6)
DD = (W >= 0)
DN = ~DD
DB = DD.astype(float)
DBN =DN.astype(float)
RES = DB*X5 + DBN*X6
