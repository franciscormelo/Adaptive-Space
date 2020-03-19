from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
from matplotlib import patches
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



el_nb = 4
fig, ax = plt.subplots()

# these next few lines are pretty important because
# otherwise your ellipses might only be displayed partly
# or may be distorted
ax.set_xlim([-5, 15])
ax.set_ylim([-5, 15])
ax.set_aspect('equal')

# first ellipse in blue
ellipse1 = create_ellipse((0, 4), (2, 4), 90)
verts1 = np.array(ellipse1.exterior.coords.xy)
patch1 = Polygon(verts1.T, color='blue', alpha=0.5)
ax.add_patch(patch1)

# second ellipse in red
ellipse2 = create_ellipse((4, 7), (2, 4), 0)
verts2 = np.array(ellipse2.exterior.coords.xy)
patch2 = Polygon(verts2.T, color='red', alpha=0.5)
ax.add_patch(patch2)

# first ellipse in blue
ellipse5 = create_ellipse((7.5, 4), (2, 4), 90)
verts5 = np.array(ellipse5.exterior.coords.xy)
patch5 = Polygon(verts5.T, color='blue', alpha=0.5)
ax.add_patch(patch5)

# second ellipse in red
ellipse4 = create_ellipse((4, 0), (2, 4), 0)
verts4 = np.array(ellipse4.exterior.coords.xy)
patch4 = Polygon(verts4.T, color='red', alpha=0.5)
ax.add_patch(patch4)

# the intersect will be outlined in black
intersect = ellipse1.intersection(ellipse2)
verts3 = np.array(intersect.exterior.coords.xy)
patch3 = Polygon(verts3.T, facecolor='none', edgecolor='black')
ax.add_patch(patch3)


print(np.amin(verts3[0]))
print(np.amax(verts3[0]))
diff = abs(np.amax(verts3[0]) - np.amin(verts3[0]))
print(diff)

diff_y = abs(np.amax(verts3[1]) - np.amin(verts3[1]))
print("Difference in y " + str(diff_y))
# compute areas and ratios
print('area of ellipse 1:', ellipse1.area)
print('area of ellipse 2:', ellipse2.area)
print('area of intersect:', intersect.area)
print('intersect/ellipse1:', intersect.area / ellipse1.area)
print('intersect/ellipse2:', intersect.area / ellipse2.area)

p = patches.Rectangle((np.amin(verts3[0]), np.amin(verts3[1])), diff, diff_y,fill=False)

ax.add_patch(p)


ellipse3 = create_ellipse((0, 4), (2, 4 - diff/2), 90)
verts3 = np.array(ellipse3.exterior.coords.xy)
patch3 = Polygon(verts3.T, color='blue', alpha=0.5)
ax.add_patch(patch3)

ellipse7 = create_ellipse((4, 7), (2, 4 - diff/2), 0)
verts7 = np.array(ellipse7.exterior.coords.xy)
patch7 = Polygon(verts7.T, color='blue', alpha=0.5)
ax.add_patch(patch7)


# first ellipse in blue
ellipse10 = create_ellipse((7.5, 4), (2, 4-diff/2), 90)
verts10 = np.array(ellipse10.exterior.coords.xy)
patch10 = Polygon(verts10.T, color='blue', alpha=0.5)
ax.add_patch(patch10)

# second ellipse in red
ellipse11 = create_ellipse((4, 0), (2, 4-diff/2), 0)
verts11 = np.array(ellipse11.exterior.coords.xy)
patch11 = Polygon(verts11.T, color='red', alpha=0.5)
ax.add_patch(patch11)



plt.show(block=False)
print("==================================================")
input("Hit Enter To Close... ")
plt.close()
