# Source: https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def params_conversion(sx, sy, angle):
    " Converts ellipses parameteres to Covarince matrix based on the orientation."

    R = np.matrix([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    S = np.matrix([[sx/2 , 0.],[0. ,sy/2]])
    T = R * S
    covariance = T * T.transpose()
    return covariance

# Our 2-dimensional distribution will be over variables X and Y
#N = 40
N = 200
X = np.linspace(-10, 10, N)
Y = np.linspace(-10, 10, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
#Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])
#Sigma = np.array([[ 1. , 0.], [0.,  1.]])

covariance = params_conversion(1,2,1)
Sigma = covariance


# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# plot using subplots
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1,projection='3d')

mu = np.array([5., 5.])
Z1= multivariate_gaussian(pos, mu, Sigma)
#normalize = cm.colors.Normalize(vmin=0, vmax=1)
#norm = normalize
Z = Z + Z1

surf = ax1.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0, antialiased=False,
                cmap=cm.coolwarm)


ax1.view_init(55,-70)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
#fig.colorbar(surf, shrink=0.5, aspect=5)

#ax2 = fig.add_subplot(2,1,2,projection='3d')
ax2 = fig.add_subplot(2,1,2)

#ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis) #fills contour lines
ax2.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis,linewidths=0.8)



#ax2.view_init(90, 270)

#ax2.grid(False)
#ax2.set_xticks([])
#ax2.set_yticks([])
#ax2.set_zticks([])
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')


plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()
