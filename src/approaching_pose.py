#! /usr/bin/env python3
'''
    File name: approaching_pose.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''

from scipy import spatial
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def approaching_area_filtering(x, y, approaching_area, contour_points):
	""" Filters the approaching area by checking the points where the cost is zero."""
	""" A point has zero cost if it is outside the personal space of all members of the group and group space."""
	cx = [j[0] for j in contour_points]
	cy = [k[1] for k in contour_points]
	polygon = Polygon(contour_points)
	
	approaching_filter = [(x, y) for x, y in zip(approaching_area[0], approaching_area[1]) if not polygon.contains(Point([x, y]))]

	return approaching_filter

