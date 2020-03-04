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
ellipse = patches.Ellipse((0,0),1,1,180)
points = [(x,y) for x,y in zip(cx, cy) if circle.contains_point([x,y])]

new = [i[0] for i in points]
print(points)
print(new)

ax.add_patch(arc1)
ax.add_patch(arc2)
ax.add_patch(circle)


#plt.plot(data[0], data[1],)
ax.set(xlim=(-2, 2), ylim=(-2, 2))
plt.show()



# [-50 50.0 0.0],[-50 -50.0 0.0] side by side

# [-148.20000000000002 32.5 -1.3997357615698351],[-230.1 9.1 -0.13608735088081467],[-148.20000000000002 -75.4 1.2893215082495184],[-97.5 -32.5 2.566782670772754]


#[-230 0.0 0.0],[230 0.0 3.14]
