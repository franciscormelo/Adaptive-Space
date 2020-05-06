#! /usr/bin/env python3
'''
    File name: space_modeling.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import math
from ellipse import *
import statistics
import sys
from shapely.geometry.point import Point
from matplotlib.patches import Polygon
from shapely import affinity
from typing import Any, Union
from matplotlib import rc

from scipy import optimize

from gaussian_modeling import plot_gaussians

SHOW_PLOT = True

# CONSTANTS
# Human Body Dimensions top view in cm
HUMAN_Y = 62.5
HUMAN_X = 37.5

# Personal Space Maximum 45 - 120 cm
PSPACEX = 80.0
PSPACEY = 60.0

# Porpotinal factor between pspace size in x and y axis
PFACTOR = PSPACEX / PSPACEY

INCREMENT = 1


def test_func(x, a, b):
    return a * np.sin(b * x)


def create_shapely_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    angle = math.degrees(angle)
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


def group_radius(persons, group_pose):
    """Computes the radius of a group."""
    sum_radius = 0
    for i in range(len(persons)):
        # average of the distance between the group members and the center of the group, o-space radius
        sum_radius = sum_radius + \
            euclidean_distance(
                persons[i][0], persons[i][1], group_pose[0], group_pose[1])
    return sum_radius / len(persons)


def euclidean_distance(x1, y1, x2, y2):
    """Euclidean distance between two points in 2D."""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


class SpaceModeling:

    def __init__(self, fh):
        """Models the personal space, group space and estimates the possibles approaching areas."""

        # Lists Initialization

        # split the file into lines
        lines = fh.read().splitlines()

        # removes blank lines
        file = list(filter(None, lines))
        n = len(file)  # Number of groups in the file

        # Lists intialization
        self.group_nb = []
        self.persons = [[] for i in range(n)]
        self.group_pose = []
        self.group_radius = []
        # Stores the personal space parameteres for each group
        self.pspace_param = [[] for i in range(n)]

        for num, string in enumerate(file):
            data = string.split("--")
            group = tuple(data[0].split(","))
            self.group_nb.append(len(group))  # Numbers of members in a group

            for person in group:
                pose = tuple(person.split(" "))
                self.persons[num].append(
                    tuple([float(pose[0][1:]), float(pose[1]), float(pose[2][:-1])]))

            self.persons[num] = tuple(self.persons[num])

            # Saves the group center i.e. o space center
            group_center = data[1].split(",")
            self.group_pose.append(tuple([float(group_center[0][1:]), float(
                group_center[1][:-1])]))  # o space center point

            # computes group radius given the center of the o-space
            radius = group_radius(self.persons[num], self.group_pose[num])
            self.group_radius.append(radius)

    def solve(self):
        """ Estimates the personal space and group space."""
        f, ax = plt.subplots(1)
        x1 = []
        y1= []

        # Iterate over groups
        for k in range(len(self.group_nb)):
            print("Modeling Group " + str(k + 1) + " ...")

            persons = self.persons[k]
            group_nb = self.group_nb[k]

            # first ellipse in blue
            ellipse1 = create_shapely_ellipse(
                (persons[0][0], persons[0][1]), (PSPACEY, PSPACEX), persons[0][2])
            verts1 = np.array(ellipse1.exterior.coords.xy)
            patch1 = Polygon(verts1.T, color='blue', alpha=0.5, fill=False)
            # ax.add_patch(patch1)

            # second ellipse in red
            ellipse2 = create_shapely_ellipse(
                (persons[1][0], persons[1][1]), (PSPACEY, PSPACEX), persons[1][2])
            verts2 = np.array(ellipse2.exterior.coords.xy)
            patch2 = Polygon(verts2.T, color='red', alpha=0.5, fill=False)
            # ax.add_patch(patch2)

            # the intersect will be outlined in black
            intersect = ellipse1.intersection(ellipse2)
            verts3 = np.array(intersect.exterior.coords.xy)
            patch3 = Polygon(verts3.T, facecolor='none', edgecolor='black')
            # ax.add_patch(patch3)

            print('area of intersect:', intersect.area)
            x1.append(persons[1][2])
            y1.append(intersect.area)

        x = np.asarray(x1)
        y = np.asarray(y1)
        params, params_covariance = optimize.curve_fit(test_func, x, y,
                                                       p0=[2, 2])
        print(params)


        plt.scatter(x, y, label='Data')
        x_test = np.linspace(0,np.max(x), 200)
        #plt.plot(x_test, test_func(x_test, params[0], params[1]),label='Fitted function')

        plt.legend(loc='best')


        #ax.set(xlim=(0, 10), ylim=(5000, 7000))
        #ax.set(xlim=(-200, 200), ylim=(-200, 200))
        plt.xlabel('angle')
        plt.ylabel('intersect area')

        if SHOW_PLOT:
            plt.tight_layout()
            plt.show(block=False)
            print("==================================================")
            input("Hit Enter To Close... ")
        plt.close()


def main():
    if len(sys.argv) > 1:
        file = "data/" + sys.argv[1]
        with open(file) as fh:
            app = SpaceModeling(fh)
            app.solve()
            fh.close()

    else:
        print("Usage: %s <filename>" % (sys.argv[0]))


if __name__ == "__main__":
    main()
