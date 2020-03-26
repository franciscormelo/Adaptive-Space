#! /usr/bin/env python3

"""
@author Francisco Melo
francisco.raposo.melo@tecnico.ulisboa.pt
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import math
from ellipse import *
import statistics
import sys
from shapely.geometry.point import Point
from shapely import affinity


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


def euclidean_distance(x1, y1, x2, y2):
    """Euclidean distance between two points in 2D."""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def draw_arrow(x, y, angle):  # angle in radians
    """Draws an arrow given a pose."""
    r = 10  # or whatever fits you
    plt.arrow(x, y, r * math.cos(angle), r * math.sin(angle),
              head_length=1, head_width=1, shape='full', color='blue')


def draw_person_top(x, y, angle, ax):
    """Draws a persons from a top view."""
    top_y = HUMAN_Y / 2
    top_x = HUMAN_X / 2
    plot_ellipse(semimaj=top_x, semimin=top_y,
                 phi=angle, x_cent=x, y_cent=y, ax=ax)


def draw_personalspace(x, y, angle, ax, sx, sy, plot_kwargs, idx):
    """Draws personal space of an inidivdual."""
    draw_arrow(x, y, angle)  # orientation arrow angle in radians
    ax.plot(x, y, 'bo', markersize=8)
    draw_person_top(x, y, angle, ax)
    ax.text(x + 3, y + 3, "$P_" + str(idx) + "$", fontsize=12)

    plot_ellipse(semimaj=sx, semimin=sy, phi=angle, x_cent=x, y_cent=y, ax=ax,
                 plot_kwargs=plot_kwargs)

    # Multiply sx and sy by 2 to get the diameter
    return patches.Ellipse((x, y), sx * 2, sy * 2, math.degrees(angle))


def approachingfiltering(personal_space, approaching_filter, idx):
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


def group_radius(persons, group_pose):
    """Computes the radius of a group."""
    sum_radius = 0
    for i in range(len(persons)):
        # average of the distance between the group members and the center of the group, o-space radius
        sum_radius = sum_radius + \
            euclidean_distance(
                persons[i][0], persons[i][1], group_pose[0], group_pose[1])
    return sum_radius / len(persons)


def ellipse_intersection(ellipse1, ellipse2):
    """Checks if two ellipses intersect."""

    return ellipse1.intersection(ellipse2)


def create_shapely_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr


def minimimum_personalspace(sx, sy):
    """Checks if the parameters are less the human dimensions."""
    if sy < HUMAN_Y / 2:  # the personal space should be at least the size of the individual
        sy = HUMAN_Y / 2

    if sx < HUMAN_X / 2:  # the personal space should be at least the size of the individual
        sx = HUMAN_X / 2

    return(sx, sy)


def parameters_computation(person1, person2, sigmax=PSPACEX, sigmay=PSPACEY):
    """Estimates the parameters of the personal space to avoid intersections."""
    # first ellipse in blue
    ellipse1 = create_shapely_ellipse(
        (person1[0], person1[1]), (sigmay, sigmax), person1[2])
    verts1 = np.array(ellipse1.exterior.coords.xy)

    # second ellipse in red
    ellipse2 = create_shapely_ellipse(
        (person2[0], person2[1]), (sigmay, sigmax),  person2[2])

    verts2 = np.array(ellipse2.exterior.coords.xy)

    intersect = ellipse1.intersection(ellipse2)

    # No intersection --> personal space input dimensions
    if intersect.is_empty:
        return (sigmax, sigmay)

    else:

        verts3 = np.array(intersect.exterior.coords.xy)

        # Maneira 2
        area1 = ellipse1.area - intersect.area
        area2 = ellipse2.area - intersect.area

        # Ellipse area area = pi * a * b

        # Quanto reduzo a cada um?? ver angulo entre estes
        sy = PSPACEY
        sx1 = area1 / (math.pi * sy)
        sx2 = area2 / (math.pi * sy)
        # alternativa
        # a1 = math.sqrt( (area1*1.2) / math.pi)
        # a2 = math.sqrt( (area2*1.2) / math.pi)
        # b = a1/1.2
        sy = sx1 / PFACTOR

        return (sx1, sy)


class SpaceModeling:

    def __init__(self, fh):
        """ """

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
        """ """
        f, ax = plt.subplots(1)

        # Iterate over groups
        for k in range(len(self.group_nb)):
            print("Modeling Group " + str(k + 1) + " ...")

            group_radius = self.group_radius[k]
            group_pose = self.group_pose[k]
            persons = self.persons[k]
            group_nb = self.group_nb[k]

            # O Space Modeling
            ax.plot(group_pose[0], group_pose[1], 'rx', markersize=8)
            plot_kwargs = {'color': 'r', 'linestyle': '-', 'linewidth': 1}
            plot_ellipse(semimaj=group_radius - HUMAN_X / 2, semimin=group_radius - HUMAN_X / 2, x_cent=group_pose[0],
                         y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

            # P Space Modeling
            plot_ellipse(semimaj=group_radius + HUMAN_X / 2, semimin=group_radius + HUMAN_X / 2, x_cent=group_pose[0],
                         y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

            # approaching circle area
            plot_kwargs = {'color': 'c', 'linestyle': ':', 'linewidth': 2}
            plot_ellipse(semimaj=group_radius, semimin=group_radius, x_cent=group_pose[0],
                         y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)
            approaching_area = plot_ellipse(semimaj=group_radius, semimin=group_radius, x_cent=group_pose[0],
                                            y_cent=group_pose[1], data_out=True)

    # Groups of 2 elements:
            if group_nb == 2:

                # Side-by-side arragement
                if round(persons[0][2]) == round(persons[1][2]):  # side-by-side

                    # sy = parameters_computation(persons)
                    sy = euclidean_distance(persons[0][0], persons[0][1], persons[1][0],
                                            persons[1][1]) / 2
                    sx = sy * PFACTOR

                    if sy > PSPACEY or sx > PSPACEX:  # if the persons are too far away from each other the personal space should be limited
                        sx = PSPACEX
                        sy = PSPACEY
                    else:
                        # Check if the parameters are less then human dimensions
                        (sx, sy) = minimimum_personalspace(sx, sy)

                # vis-a-vis arrangement
                elif abs(round(persons[0][2] + math.pi, 2)) == abs(round(persons[1][2], 2)):

                    sx = euclidean_distance(persons[0][0], persons[0][1], persons[1][0],
                                            persons[1][1]) / 2
                    sy = sx / PFACTOR

                    if sy > PSPACEY or sx > PSPACEX:  # if the persons are too far away from each other the personal space should be limited
                        sx = PSPACEX
                        sy = PSPACEY
                    else:
                        # Check if the parameters are less then human dimensions
                        (sx, sy) = minimimum_personalspace(sx, sy)

                else:  # other arrangements

                    (sx, sy) = parameters_computation(
                        persons[0], persons[1], sigmax=PSPACEX, sigmay=PSPACEY)
                    # Check if the parameters are less then human dimensions
                    (sx, sy) = minimimum_personalspace(sx, sy)


# Groups of > 2 elements:
            else:  # The typical arragement  of a group of more than 2 persons is tipically circular

                sx = PSPACEX
                sy = PSPACEY

                for i in range(len(persons) - 1):
                    w = i
                    for j in range(w + 1, len(persons)):

                        (sx, sy) = parameters_computation(
                            persons[i], persons[j], sigmax=sx, sigmay=sy)

            # Check if the parameters are less then human dimensions
            (sx, sy) = minimimum_personalspace(sx, sy)
                
            if sx > PSPACEX or sy > PSPACEY:  # if the persons are too far away from each other the personal space should be limited
                sx = PSPACEX
                sy = PSPACEY

    ###############################################################################
        # Possible approaching area computation and personal space ploting
            plot_kwargs = {'color': 'g', 'linestyle': '-', 'linewidth': 0.8}

            approaching_filter = approaching_area

            for idx, person in enumerate(persons, start=1):

                personal_space = draw_personalspace(
                    person[0], person[1], person[2], ax, sx, sy, plot_kwargs, idx)

                # Approaching Area filtering - remove points that are inside the personal space of a person
                approaching_filter = approachingfiltering(
                    personal_space, approaching_filter, idx)

            # possible approaching positions
            approaching_x = [j[0] for j in approaching_filter]
            approaching_y = [k[1] for k in approaching_filter]

            ax.plot(approaching_x, approaching_y, 'c.',  markersize=5)

            # x = [item[0] for item in persons]
            # y = [item[1] for item in persons]
            # ax.plot([persons[0][0],persons[1][0]],[persons[0][1],persons[1][1]])
            # ax.plot(x, y, 'g')

        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        plt.savefig('destination_path.eps', format='eps')

        if SHOW_PLOT == True:
            plt.show(block=False)
            print("==================================================")
            input("Hit Enter To Close... ")
            plt.close()


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fh:
            app = SpaceModeling(fh)
            app.solve()

            fh.close()
    else:
        print("Usage: %s <filename>" % (sys.argv[0]))


if __name__ == "__main__":
    main()
