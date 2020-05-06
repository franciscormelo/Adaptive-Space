#! /usr/bin/env python3
'''
    File name: approaching_pose.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''

import math

from scipy import spatial

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def euclidean_distance(x1, y1, x2, y2):
    """Euclidean distance between two points in 2D."""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def approachingfiltering_ellipses(personal_space, approaching_filter, idx):
    """Filters the approaching area."""
    # Approaching Area filtering - remove points tha are inside the personal space of a person
    if idx == 1:
        approaching_filter = [(x, y) for x, y in zip(
            approaching_filter[0], approaching_filter[1]) if not personal_space.contains_point([x, y])]
    else:
        cx = [j[0] for j in approaching_filter]
        cy = [k[1] for k in approaching_filter]
        approaching_filter = [(x, y) for x, y in zip(
            cx, cy) if not personal_space.contains_point([x, y])]
    return approaching_filter


def approaching_area_filtering(x, y, approaching_area, contour_points):
    """ Filters the approaching area by checking the points where the cost is zero."""
    """ A point has zero cost if it is outside the personal space of all members of the group and group space."""
    cx = [j[0] for j in contour_points]
    cy = [k[1] for k in contour_points]
    polygon = Polygon(contour_points)

    approaching_filter = [(x, y) for x, y in zip(
        approaching_area[0], approaching_area[1]) if not polygon.contains(Point([x, y]))]

    return approaching_filter


def approaching_pose(robot_pose, approaching_area, group_center):
    """Chooses the nearest center point to the robot from the multiple approaching area."""
    min_dis = 0
    for i in range(len(approaching_area)):
        if i == 0:
            min_dis = euclidean_distance(
                robot_pose[0], robot_pose[1], approaching_area[i][0], approaching_area[i][1])
            min_idx = 0
        else:
            dis = euclidean_distance(
                robot_pose[0], robot_pose[1], approaching_area[i][0], approaching_area[i][1])

            if dis < min_dis:
                min_dis = dis
                min_idx = i
    goal_x = approaching_area[min_idx][0]
    goal_y = approaching_area[min_idx][1]
    orientation = math.atan2(
        group_center[1] - goal_y, group_center[0] - goal_x)

    goal_pose = [goal_x, goal_y, orientation]

    return goal_pose
