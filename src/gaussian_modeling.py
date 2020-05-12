#! /usr/bin/env python3
'''
    File name: gaussian_modeling.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
from ellipse import *
from matplotlib import rc

from scipy.stats import multivariate_normal

from approaching_pose import *



# CONSTANTS
# Human Body Dimensions top view in cm
HUMAN_Y = 45
HUMAN_X = 20

# Relation between personal frontal space and back space
BACK_FACTOR = 1.3

# APPROACHING LEVEL
LEVEL = 1


def plot_person(x, y, angle, ax, plot_kwargs):
    """ Plots a person from a top view."""
    r = 10  # or whatever fits you
    ax.arrow(x, y, r * math.cos(angle), r * math.sin(angle),
             head_length=1, head_width=1, shape='full', color='blue')

    ax.plot(x, y, 'bo', markersize=8)

    top_y = HUMAN_Y / 2
    top_x = HUMAN_X / 2
    plot_ellipse(semimaj=top_x, semimin=top_y,
                 phi=angle, x_cent=x, y_cent=y, ax=ax)


def plot_group(group_pose,group_radius, pspace_radius,ospace_radius, ax):
    """Plots the group o space, p space and approaching circle area. """
    # O Space Modeling
    ax.plot(group_pose[0], group_pose[1], 'rx', markersize=8)
    plot_kwargs = {'color': 'r', 'linestyle': '-', 'linewidth': 1}

    
    plot_ellipse(semimaj=ospace_radius, semimin=ospace_radius, x_cent=group_pose[0],
                 y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

    # P Space Modeling
    
    plot_ellipse(semimaj=pspace_radius, semimin=pspace_radius, x_cent=group_pose[0],
                 y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

    # approaching circle area
    plot_kwargs = {'color': 'c', 'linestyle': ':', 'linewidth': 2}
    plot_ellipse(semimaj=group_radius, semimin=group_radius, x_cent=group_pose[0],
                 y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)
    approaching_area = plot_ellipse(semimaj=group_radius, semimin=group_radius, x_cent=group_pose[0],
                                    y_cent=group_pose[1], data_out=True)
    return approaching_area


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    # Adapted from: https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    n = mu.shape[0]  # Dimension
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi)**n * Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def asymmetric_gaussian(pos, mu, Sigma, orientation, center, N, Sigma_back):
    """ Computes an asymmetric  2D gaussian function using a function for the frontal part and another one for the back part"""
    Z1 = np.zeros([N, N])
    Z2 = np.zeros([N, N])
    angle = orientation + math.pi / 2

    # # Based on Kirby phd thesis
    cond = np.arctan2(pos[:, :, 1] - center[1], pos[:, :,
                                                    0] - center[0]) - orientation + (math.pi / 2)

    # Front gaussian
    #aux1 = (cond + np.pi) % (2 * np.pi) - np.pi > 0
    # Compute the nor- malized angle of the line
    aux1 = np.arctan2(np.sin(cond), np.cos(cond)) > 0
    pos1 = pos[:, :][aux1]
    Z1[aux1] = multivariate_gaussian(pos1, mu, Sigma)

    # Back Gaussian
    #aux2 = (cond + np.pi) % (2 * np.pi) - np.pi <= 0
    # Compute the nor- malized angle of the line
    aux2 = np.arctan2(np.sin(cond), np.cos(cond)) <= 0
    pos2 = pos[:, :][aux2]
    Z2[aux2] = multivariate_gaussian(pos2, mu, Sigma_back)

    # Normalization
    A1 = 1 / Z1.max()
    Z1 = A1 * Z1

    A2 = 1 / Z2.max()
    Z2 = A2 * Z2
    return Z1 + Z2


def params_conversion(sx, sy, angle):
    """ Converts ellipses parameteres to Covarince matrix based on the orientation."""
    # https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

    R = np.matrix([[math.cos(angle), -math.sin(angle)],
                   [math.sin(angle), math.cos(angle)]])
    S = np.matrix([[sx / 2, 0.], [0., sy / 2]])
    T = R * S
    covariance = T * T.transpose()

    return covariance


def draw_arrow(x, y, angle):  # angle in radians
    """Draws an arrow given a pose."""
    r = 10  # or whatever fits you
    plt.arrow(x, y, r * math.cos(angle), r * math.sin(angle),
              head_length=1, head_width=1, shape='full', color='black')


def plot_robot(pose, ax):
    x = pose[0]
    y = pose[1]
    angle = pose[2]

    """Draws a robot from a top view."""
    top_y = HUMAN_Y / 2
    top_x = HUMAN_Y / 2
    plot_kwargs = {'color': 'black', 'linestyle': '-', 'linewidth': 1}
    plot_ellipse(semimaj=top_x, semimin=top_y,
                 phi=angle, x_cent=x, y_cent=y, ax=ax, plot_kwargs=plot_kwargs)

    draw_arrow(x, y, angle)  # orientation arrow angle in radians
    ax.plot(x, y, 'o', color='black', markersize=5)
    return


def plot_gaussians(persons, group_data, idx, ellipse_param, N=200, show_group_space=True):
    """ Plots surface and contour of 2D Gaussian function given ellipse parameters."""

    group_radius = group_data['group_radius'][idx]
    pspace_radius = group_data['pspace_radius'][idx]
    ospace_radius = group_data['ospace_radius'][idx]
    group_pos = group_data['group_pose'][idx]
    # Initial Gaussians amplitude
    A = 1

    # Gets the values of x and y of all the persons
    x = [item[0] for item in persons]
    y = [item[1] for item in persons]

    # Gets the coordinates of a windows around the group
    xmin = min(x) - 150
    xmax = max(x) + 150
    ymin = min(y) - 150
    ymax = max(y) + 150

    X_lin = np.linspace(xmin, xmax, N)
    Y_lin = np.linspace(ymin, ymax, N)
    X = X_lin
    Y = Y_lin
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = np.empty([N, N])

    # plot using subplots
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    ax2 = fig.add_subplot(1, 2, 1)

    plot_kwargs = {'color': 'g', 'linestyle': '-', 'linewidth': 0.8}
    # Personal Space as gaussian for each person in the group

    for person in persons:
        Z1 = None
        mu = np.array([person[0], person[1]])
        Sigma = params_conversion(
            ellipse_param[0], ellipse_param[1], person[2])

        Sigma_back = params_conversion(
            ellipse_param[0] / BACK_FACTOR, ellipse_param[1], person[2])

        # The distribution on the variables X, Y packed into pos.
        Z1 = asymmetric_gaussian(
            pos, mu, Sigma, person[2], (person[0], person[1]), N, Sigma_back)

        #Z1 = multivariate_normal(mu, Sigma).pdf(pos)
        #Z = Z1

        # Z matrix only updates the values where Z1 > Z
        cond = Z1 > Z
        Z[cond] = Z1[cond]

        plot_person(person[0], person[1], person[2], ax2, plot_kwargs)

    approaching_area = plot_group(group_pos, group_radius,pspace_radius, ospace_radius, ax2)

    show_group_space = True
    if show_group_space:
        Z1 = None
        mu = np.array([group_pos[0], group_pos[1]])
        ospace_radius = group_radius - HUMAN_X / 2

        Sigma = params_conversion(ospace_radius, ospace_radius, 0)

        Z1 = A * multivariate_gaussian(pos, mu, Sigma)
        Z1 = multivariate_normal(mu, Sigma).pdf(pos)
        # Normalization

        A1 = 1 / Z1.max()
        Z1 = A1 * Z1

        # Z matrix only updates the values where Z1 > Z
        cond = Z1 > Z
        Z[cond] = Z1[cond]

    surf = ax1.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=1,
                            antialiased=False, cmap="jet")

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax1.set_xlabel(r'$x$ $[cm]$')
    ax1.set_ylabel(r'$y$ $[cm]$')
    ax1.set_zlabel(r'Cost')

    cs = ax2.contour(X, Y, Z, cmap="jet", linewidths=0.8, levels=10)
    #cs = ax2.contour(X, Y, Z, cmap="jet", linewidths=0.8)
    fig.colorbar(cs)

    # Approaching Area filtering - remove points that are inside the personal space of a person
    approaching_filter = approaching_area_filtering(
        X_lin, Y_lin, approaching_area, cs.allsegs[LEVEL][0])
    x_approach = [j[0] for j in approaching_filter]
    y_approach = [k[1] for k in approaching_filter]
    ax2.plot(x_approach, y_approach, 'c.', markersize=5)

   # robot_pose = [0, 0, 3.14]
    # Plots robot from top view
   # plot_robot(robot_pose, ax2)
    #ax2.annotate("Robot_Initial", (robot_pose[0], robot_pose[1]))

    # Estimates the goal pose for the robot to approach the group
    #goal_pose = approaching_pose(robot_pose, approaching_filter, group_pos)
    #print("Goal Pose (cm,cm,rad) = " + str(goal_pose))
    #plot_robot(goal_pose, ax2)
    #ax2.annotate("Robot_Goal", (goal_pose[0], goal_pose[1]))
    ax2.set_xlabel(r'$x$ $[cm]$')
    ax2.set_ylabel(r'$y$ $[cm]$')
    ax2.set_aspect(aspect=1)
    fig.tight_layout()
    plt.show(block=False)
    print("==================================================")
    input("Hit Enter To Close... ")
    surf.remove()
    plt.clf()
    plt.close()
