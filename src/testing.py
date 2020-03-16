#! /usr/bin/env python3


# arc instead of ellipse testing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import math
from ellipse import *
import statistics
import sys

f, ax = plt.subplots(1)
arc1 = patches.Arc((0,0),1,2,0., 0.0, 180.0)
arc2= patches.Arc((0,0),1,1,180, 0, 180.0)



circle = patches.Circle((1,1), radius =  1, fill = False)


cx = [1 ,2, 3]
cy = [1 ,2 ,3]
sx = 1
sy = 2
x0 = 0
y0 = 0
alpha = 45
ellipse = patches.Ellipse((x0,y0),sx*2,sy*2,alpha, fill = False)
points = [(x,y) for x,y in zip(cx, cy) if circle.contains_point([x,y])]

new = [i[0] for i in points]
# print(points)
# print(new)

#ax.add_patch(arc1)
#ax.add_patch(arc2)
#ax.add_patch(circle)
ax.add_patch(ellipse)

alpha = math.radians(alpha)
angle = math.pi
xprime = sx * math.cos(angle) + x0
yprime = sy * math.sin(angle) + y0

#
x = math.cos(alpha) * xprime - math.sin(alpha) * yprime
y = math.sin(alpha) * xprime + math.cos(alpha) * yprime
plt.plot(x, y, 'bo', markersize=8)


#plt.plot(data[0], data[1],)
ax.set(xlim=(-2, 2), ylim=(-2, 2))
plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()



# [-50 50.0 0.0],[-50 -50.0 0.0] side by side center - (0, 0)

# [-148.20000000000002 32.5 -1.3997357615698351],[-230.1 9.1 -0.13608735088081467],[-148.20000000000002 -75.4 1.2893215082495184],[-97.5 -32.5 2.566782670772754] center -


#[-230 0.0 0.0],[230 0.0 3.14] center - (0, 0)
