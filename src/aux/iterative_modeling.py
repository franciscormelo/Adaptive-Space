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

def pspace_intersection(person1, person2, sigmax, sigmay):
    """ """

    ellipse1 = create_shapely_ellipse(
        (person1[0], person1[1]), (sigmax, sigmay), person1[2])

    # second ellipse in red
    ellipse2 = create_shapely_ellipse(
        (person2[0], person2[1]), (sigmax, sigmay), person2[2])

    return ellipse_intersection(ellipse1, ellipse2)


fig, ax = plt.subplots()

# these next few lines are pretty important because
# otherwise your ellipses might only be displayed partly
# or may be distorted
ax.set_xlim([400, 800])
ax.set_ylim([0 ,600])
ax.set_aspect('equal')
sy = 43.95000000000063
sx = 58.60000000000083


# first ellipse in blue
ellipse1 = create_ellipse((600, 320), (sx, sy), -90)
verts1 = np.array(ellipse1.exterior.coords.xy)
patch1 = Polygon(verts1.T, color='blue', alpha=0.5)
ax.add_patch(patch1)

# second ellipse in red
ellipse2 = create_ellipse((670, 250), (sx, sy), -180)
verts2 = np.array(ellipse2.exterior.coords.xy)
patch2 = Polygon(verts2.T, color='red', alpha=0.5)
ax.add_patch(patch2)

# the intersect will be outlined in black
intersect = ellipse1.intersection(ellipse2)
verts3 = np.array(intersect.exterior.coords.xy)
patch3 = Polygon(verts3.T, facecolor='none', edgecolor='black')
ax.add_patch(patch3)


diff = 0.1
print('area of intersect:', intersect.area)

while (intersect.area) != 0:
    if sx >= 0 and sy >= 0:
        sx = sx - diff
        #sy = sy - diff
        sy = sx /(80/60)


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

plt.tight_layout()
plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()
