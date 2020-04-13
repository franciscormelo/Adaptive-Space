from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
import numpy as np


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


fig, ax = plt.subplots()

# these next few lines are pretty important because
# otherwise your ellipses might only be displayed partly
# or may be distorted
ax.set_xlim([-1000, 1000])
ax.set_ylim([-1000, 1000])
ax.set_aspect('equal')



# first ellipse in blue
ellipse1 = create_ellipse((600, 320), (60, 80), -90)
verts1 = np.array(ellipse1.exterior.coords.xy)
patch1 = Polygon(verts1.T, color='blue', alpha=0.5)
ax.add_patch(patch1)

# second ellipse in red
ellipse2 = create_ellipse((670, 250), (60, 80), -180)
verts2 = np.array(ellipse2.exterior.coords.xy)
patch2 = Polygon(verts2.T, color='red', alpha=0.5)
ax.add_patch(patch2)

# the intersect will be outlined in black
intersect = ellipse1.intersection(ellipse2)
verts3 = np.array(intersect.exterior.coords.xy)
patch3 = Polygon(verts3.T, facecolor='none', edgecolor='black')
ax.add_patch(patch3)


diff = 1
print('area of intersect:', intersect.area)
sx = 80
sy = 60
while (intersect.area) != 0:
    if sx >= 0 and sy >= 0:
        sx = sx - diff
        #sy = sy - diff
        st = sx / 1.5


        # first ellipse in blue
        ellipse1 = create_ellipse((600, 320), (sx, sy), -90)

        # second ellipse in red
        ellipse2 = create_ellipse((670, 250), (sx, sy), -180)

        intersect = ellipse1.intersection(ellipse2)
        print("Sx = " + str(sx) + " Sy = " + str(sy))
        print('area of intersect:', intersect.area)


# first ellipse in blue
verts4 = np.array(ellipse1.exterior.coords.xy)
patch4 = Polygon(verts4.T, color='blue', alpha=0.5)
ax.add_patch(patch4)

# second ellipse in red
verts5 = np.array(ellipse2.exterior.coords.xy)
patch5 = Polygon(verts5.T, color='red', alpha=0.5)
ax.add_patch(patch5)


plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()
