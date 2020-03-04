#! /usr/bin/env python3

"""Version ellipse function
    default unit is cm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import math
from ellipse import *
import statistics
import sys

# CONSTANTS
# Human Body Dimensions top view in cm
HUMAN_Y = 62.5
HUMAN_X = 37.5

def euclidean_distance(x1, y1, x2, y2):
    """ """
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def draw_arrow(x, y, angle):  # angle in radians
    """ """
    r = 10  # or whatever fits you
    plt.arrow(x, y, r * math.cos(angle), r * math.sin(angle), head_length=1, head_width=1, shape='full', color='blue')


def draw_person_top(x, y, angle, ax):
    """ """
    top_y = HUMAN_Y / 2
    top_x = HUMAN_X / 2
    plot_ellipse(semimaj=top_x, semimin=top_y, phi=angle, x_cent=x, y_cent=y, ax=ax)

def draw_personalspace(x, y, angle, ax, sx, sy, plot_kwargs, idx):
    """ """
    draw_arrow(x, y, angle)  # orientation arrow angle in radians
    ax.plot(x, y, 'bo', markersize=8)
    draw_person_top(x, y, angle, ax)
    ax.text(x + 3, y + 3, "$P_" + str(idx) + "$", fontsize=12)

    plot_ellipse(semimaj=sx, semimin=sy, phi=angle, x_cent=x, y_cent=y, ax=ax,
                 plot_kwargs=plot_kwargs)

    return patches.Ellipse((x,y),sx*2,sy*2,math.degrees(angle)) # Multiply sx and sy by 2 to get the diameter

def approachingfiltering(personal_space, approaching_filter, idx):
    # Approaching Area filtering - remove points tha are inside the personal space of a person
    if idx == 1:
        approaching_filter = [(x,y) for x,y in zip(approaching_filter[0], approaching_filter[1]) if not personal_space.contains_point([x,y])]
    else:
        cx = [j[0] for j in approaching_filter]
        cy = [k[1] for k in approaching_filter]
        approaching_filter = [(x,y) for x,y in zip(cx, cy) if not personal_space.contains_point([x,y])]
    return approaching_filter


class SpaceModeling:

    def __init__(self, fh):
        """ """
        # split the file into lines
        lines = fh.read().splitlines()

        # removes blank lines
        file = list(filter(None, lines))

        for string in file:
            group = tuple(string.split(","))
            self.group_nb = len(group) # Numbers of members in a group

            self.persons = []
            for person in group:
                pose = tuple(person.split(" "))
                self.persons.append(tuple([float(pose[0][1:]), float(pose[1]), float(pose[2][:-1])]))

            self.persons = tuple(self.persons)
            self.group_pose = (-157.0, -13.0)  # o space center point

            # compute group radius given the center of the o-space
            self.group_radius = self.group_radius()

    def group_radius(self):
        """ """
        sum_radius = 0
        for i in range(len(self.persons)):
            # average of the distance between the group members and the center of the group, o-space radius
            sum_radius = sum_radius + euclidean_distance(self.persons[i][0], self.persons[i][1], self.group_pose[0], self.group_pose[1])
        return sum_radius / len(self.persons)


    def solve(self):
        """ """
        # (pos_x, pos_y, angles)
        # p_pose = ((1.,2.,math.pi/6),(2.,3.,-math.pi/2),(3.,2.,5*math.pi/6),(2.,1.,math.pi/2)) # 4 ciruclar arragement proximos
        # p_pose = ((1.5,2.,math.pi/6),(2.,2.5,-math.pi/2),(2.5,2.,5*math.pi/6),(2.,1.5,math.pi/2)) # 4 ciruclar arragement proximos
        # p_pose = ((2,2.,math.pi/2),(3.,2.,math.pi/2)) # vis a vis

        f, ax = plt.subplots(1)

        # O Space Modeling
        ax.plot(self.group_pose[0], self.group_pose[1], 'rx', markersize=8)
        plot_kwargs = {'color': 'r', 'linestyle': '-', 'linewidth': 1}
        plot_ellipse(semimaj=self.group_radius - HUMAN_X/2, semimin=self.group_radius - HUMAN_X/2, x_cent=self.group_pose[0],
                     y_cent=self.group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

        # p Space Modeling
        plot_ellipse(semimaj=self.group_radius + HUMAN_X/2, semimin=self.group_radius + HUMAN_X/2, x_cent=self.group_pose[0],
                     y_cent=self.group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

        # approaching circle area
        plot_kwargs = {'color': 'c', 'linestyle': ':', 'linewidth': 2}
        plot_ellipse(semimaj=self.group_radius, semimin=self.group_radius, x_cent=self.group_pose[0],
                            y_cent=self.group_pose[1], ax=ax, plot_kwargs=plot_kwargs)
        approaching_area = plot_ellipse(semimaj=self.group_radius, semimin=self.group_radius, x_cent=self.group_pose[0],
                            y_cent=self.group_pose[1],data_out = True)

        # compute mean distance between group members
        d_sum = 0
        for i in range(len(self.persons) - 1):
            # average of the distance between group members
            d_sum = d_sum + euclidean_distance(self.persons[i][0], self.persons[i][1], self.persons[i + 1][0],
                                              self.persons[i + 1][1])
        d_mean = d_sum / len(self.persons)
###############################################################################
        # variar a maneira como e calculado tendo em conta o tipo de grupo

        # Scaling factors for personal space
        sx = d_mean # radius in x
        sy = sx / 1.5 # radius in y
        # por um limite!!!!
        # compute personal space por each person

###############################################################################
        idx = 1
        plot_kwargs = {'color': 'g', 'linestyle': '-', 'linewidth': 0.8}

        approaching_filter = approaching_area

        for person in self.persons:

            personal_space = draw_personalspace(person[0], person[1], person[2], ax, sx, sy, plot_kwargs, idx)

            # Approaching Area filtering - remove points tha are inside the personal space of a person
            approaching_filter = approachingfiltering(personal_space, approaching_filter, idx)
            idx = idx + 1

        # possible approaching positions
        approaching_x = [j[0] for j in approaching_filter]
        approaching_y = [k[1] for k in approaching_filter]


        ax.plot(approaching_x, approaching_y, 'c.',  markersize = 5)
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        plt.savefig('destination_path.eps', format='eps')
        plt.show()


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
