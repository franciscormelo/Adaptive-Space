#! /usr/bin/env python3
'''Version using gaussian function'''
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
import statistics
import sys


# From ellipse parameters to covariance matrix
person = [1,1,0]
x, y = np.mgrid[-300:300:0.1, -300:300:0.1]
position = np.empty(x.shape + (2,))
position[:, :, 0] = x
position[:, :, 1] = y

Sx = 1
Sy = 2
A = 1

R = np.matrix([[math.cos(person[2]),-math.sin(person[2])],[math.sin(person[2]),math.cos(person[2])]])
S = np.matrix([[Sx , 0],[0 ,Sy]])
T = R * S


covariance = T * T.transpose()


# From ellipse parameters to covariance matrix

z = A * multivariate_normal([person[0],person[1]], covariance.tolist()).pdf(position)
plt.contour(x, y, z)




plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()
