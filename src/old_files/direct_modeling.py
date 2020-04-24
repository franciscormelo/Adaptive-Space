from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
from matplotlib import patches
import numpy as np
import math


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


a = 4
b = 2
c1 = (2, 4)
c2 = (4, 7)


el_nb = 4
fig, ax = plt.subplots()
fig2 = plt.subplot()

# these next few lines are pretty important because
# otherwise your ellipses might only be displayed partly
# or may be distorted
ax.set_xlim([-5, 15])
ax.set_ylim([-5, 15])
ax.set_aspect('equal')

# first ellipse in blue
ellipse1 = create_ellipse(c1, (2, 4), 90)
verts1 = np.array(ellipse1.exterior.coords.xy)
patch1 = Polygon(verts1.T, color='blue', alpha=0.5, fill=False)
ax.add_patch(patch1)

# second ellipse in red
ellipse2 = create_ellipse(c2, (2, 4), 0)
verts2 = np.array(ellipse2.exterior.coords.xy)
patch2 = Polygon(verts2.T, color='red', alpha=0.5, fill=False)
ax.add_patch(patch2)


# the intersect will be outlined in black
intersect = ellipse1.intersection(ellipse2)
verts3 = np.array(intersect.exterior.coords.xy)
patch3 = Polygon(verts3.T, facecolor='none', edgecolor='black')
ax.add_patch(patch3)
###############################################
# Maneira 1 distancia
# print(np.amin(verts3[0]))
# print(np.amax(verts3[0]))
# diff = abs(np.amax(verts3[0]) - np.amin(verts3[0]))
# print(diff)
#
# diff_y = abs(np.amax(verts3[1]) - np.amin(verts3[1]))
#
#
# print("Difference in y " + str(diff_y))
#######################################################

# compute areas and ratios
print('area of ellipse 1:', ellipse1.area)
print('area of ellipse 2:', ellipse2.area)
print('area of intersect:', intersect.area)
print('intersect/ellipse1:', intersect.area / ellipse1.area)
print('intersect/ellipse2:', intersect.area / ellipse2.area)


######################################################
# Maneira 2 area
area1 = ellipse1.area - intersect.area
area2 = ellipse2.area - intersect.area
# ver quanto tirar de area a cada uma
porpotion = 2
# Ellipse area area = pi * a * b
# por um factor que tem conta o angulo

a1 = area1 / (math.pi * b)
a2 = area2 / (math.pi * b)
# alternativa
# a1 = math.sqrt( (area1*1.2) / math.pi)
# a2 = math.sqrt( (area2*1.2) / math.pi)
#b = a1/1.2

##################################################


# area ellipse
print("Maneira 2 " + str(a1))
print("Maneira 2 " + str(a2))

#print("Maneira 1 " + str(a-(diff/2)))
#m1 = a-(diff/2)

#p = patches.Rectangle((np.amin(verts3[0]), np.amin(verts3[1])), diff, diff_y,fill=False)

# ax.add_patch(p)


ellipse3 = create_ellipse(c1, (b, a1), 90)
verts3 = np.array(ellipse3.exterior.coords.xy)
patch3 = Polygon(verts3.T, color='blue', alpha=0.5)
ax.add_patch(patch3)

ellipse7 = create_ellipse(c2, (b, a2), 0)
verts7 = np.array(ellipse7.exterior.coords.xy)
patch7 = Polygon(verts7.T, color='blue', alpha=0.5)
ax.add_patch(patch7)


plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()
