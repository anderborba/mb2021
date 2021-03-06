# -*- coding: utf-8 -*-
"""bfgs_exemple.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x2gHILXz4JfCHYYF-TkXibw3IlR8OFMd
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import math
#import os
import numpy as np
from scipy.optimize import minimize
#
def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
#
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der
#
x0 = np.array([1.3, 0.7])
#res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
#               options={'disp': True})
res = minimize(rosen, x0, method='BFGS',
               options={'disp': True})
#
print(res.x)

plt.rc('text', usetex=False)
fig = plt.figure()
ax = fig.gca(projection='3d')
xx = np.arange(-2, 2, 0.1)
yy = np.arange(-3, 1, 0.1)
x1, y1 = np.meshgrid(yy, xx)
dimx = xx.shape
dimy = yy.shape
n = dimx[0]
m = dimy[0]
s = (n, m)
z = np.zeros(s)
x = np.zeros(2)
for i in range(0, n):
  for j in range(0, m):
    x[0] = xx[i]
    x[1] = yy[j]
    z[i][j] = rosen(x)
surf =ax.plot_surface(x1, y1, z, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock function')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

"""# Nova seção"""

print(z)