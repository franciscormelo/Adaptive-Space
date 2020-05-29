#! /usr/bin/env python3
'''
    File name: gaussian_comparison.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''
import numpy as np
import matplotlib.pyplot as plt
import math
from ellipse import plot_ellipse

from scipy.stats import multivariate_normal

from approaching_pose import zones_center, approaching_heuristic, approaching_area_filtering


# CONSTANTS
# Human Body Dimensions top view in cm
HUMAN_Y = 45
HUMAN_X = 20

# Relation between personal frontal space and back space
BACK_FACTOR = 1.3

# APPROACHING LEVEL
LEVEL = 1

# DSZ PARAMETERS in cm
F_PSPACEX = 54
F_PSPACEY = 45
# F_PSPACEX = 80.0
# F_PSPACEY = 60.0

# F_PSPACEX = 120
# F_PSPACEY = 110


def plot_person(x, y, angle, ax, plot_kwargs):
    """ Plots a person from a top view."""
    draw_arrow(x, y, angle, ax)

    ax.plot(x, y, 'bo', markersize=8)

    top_y = HUMAN_Y / 2
    top_x = HUMAN_X / 2
    plot_ellipse(semimaj=top_x, semimin=top_y,
                 phi=angle, x_cent=x, y_cent=y, ax=ax)


def plot_group(group_pose, group_radius, pspace_radius, ospace_radius, ax):
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


def multivariate_gaussian(pos, mu, sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    # Adapted from: https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    n = mu.shape[0]  # Dimension
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi)**n * sigma_det)

    # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def asymmetric_gaussian(pos, mu, sigma, orientation, center, N, sigma_back):
    """ Computes an asymmetric  2D gaussian function using a function for the frontal part and another one for the back part"""
    Z1 = np.zeros([N, N])
    Z2 = np.zeros([N, N])

    # # Based on Kirby phd thesis
    cond = np.arctan2(pos[:, :, 1] - center[1], pos[:, :,
                                                    0] - center[0]) - orientation + (math.pi / 2)

    # Front gaussian
    #aux1 = (cond + np.pi) % (2 * np.pi) - np.pi > 0
    # Compute the nor- malized angle of the line
    aux1 = np.arctan2(np.sin(cond), np.cos(cond)) > 0
    pos1 = pos[:, :][aux1]
    Z1[aux1] = multivariate_gaussian(pos1, mu, sigma)

    # Back Gaussian
    #aux2 = (cond + np.pi) % (2 * np.pi) - np.pi <= 0
    # Compute the nor- malized angle of the line
    aux2 = np.arctan2(np.sin(cond), np.cos(cond)) <= 0
    pos2 = pos[:, :][aux2]
    Z2[aux2] = multivariate_gaussian(pos2, mu, sigma_back)

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


def draw_arrow(x, y, angle, ax):  # angle in radians
    """Draws an arrow given a pose."""
    r = 8  # or whatever fits you
    ax.arrow(x, y, r * math.cos(angle), r * math.sin(angle),
             head_length=1, head_width=1, shape='full', color='black')


def plot_robot(pose, ax):
    """Draws a robot from a top view."""
    x = pose[0]
    y = pose[1]
    angle = pose[2]

    top_y = HUMAN_Y / 2
    top_x = HUMAN_Y / 2
    plot_kwargs = {'color': 'black', 'linestyle': '-', 'linewidth': 1}
    plot_ellipse(semimaj=top_x, semimin=top_y,
                 phi=angle, x_cent=x, y_cent=y, ax=ax, plot_kwargs=plot_kwargs)

    draw_arrow(x, y, angle, ax)  # orientation arrow angle in radians
    ax.plot(x, y, 'o', color='black', markersize=5)


def plot_gaussians(persons, group_data, idx, ellipse_param, N=200, show_group_space=True, plot=True):
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

    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)

    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.zeros(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # plot using subplots
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    plot_kwargs = {'color': 'g', 'linestyle': '-', 'linewidth': 0.8}
    # Personal Space as gaussian for each person in the group

    Z_F = np.zeros([N, N])

    for person in persons:
        sigma = None
        sigma_back = None
        Z1 = None
        mu = np.array([person[0], person[1]])

        sigma = params_conversion(
            ellipse_param[0], ellipse_param[1], person[2])

        sigma_back = params_conversion(
            ellipse_param[0] / BACK_FACTOR, ellipse_param[1], person[2])

        # The distribution on the variables X, Y packed into pos.
        Z1 = asymmetric_gaussian(
            pos, mu, sigma, person[2], (person[0], person[1]), N, sigma_back)

        # Z matrix only updates the values where Z1 > Z
        cond = None
        cond = Z1 > Z_F
        Z_F[cond] = Z1[cond]

        plot_person(person[0], person[1], person[2], axs[0], plot_kwargs)

    F_approaching_area = plot_group(
        group_pos, group_radius, pspace_radius, ospace_radius, axs[0])

    show_group_space = True
    sigma = None
    if show_group_space:
        Z1 = None
        mu = np.array([group_pos[0], group_pos[1]])

        sigma = params_conversion(ospace_radius, ospace_radius, 0)

        Z1 = A * multivariate_gaussian(pos, mu, sigma)
        Z1 = multivariate_normal(mu, sigma).pdf(pos)
        # Normalization

        A1 = 1 / Z1.max()
        Z1 = A1 * Z1

        # Z matrix only updates the values where Z1 > Z
        cond = None
        cond = Z1 > Z_F
        Z_F[cond] = Z1[cond]

    cs1 = axs[0].contour(X, Y, Z_F, cmap="jet", linewidths=0.8, levels=10)

    F_approaching_filter, F_approaching_zones, F_limit_points = approaching_area_filtering(
        F_approaching_area, cs1.allsegs[LEVEL][0])
    F_approaching_filter, F_approaching_zones = approaching_heuristic(
        group_radius, pspace_radius, group_pos, F_approaching_filter, cs1.allsegs[LEVEL][0], F_approaching_zones)
    F_x_approach = [j[0] for j in F_approaching_filter]
    F_y_approach = [k[1] for k in F_approaching_filter]

    F_approaching_perimeter = (
        len(F_x_approach) * 2 * math.pi * group_radius) / len(F_approaching_area[0])
    axs[0].plot(F_x_approach, F_y_approach, 'c.', markersize=5)

    F_center_x, F_center_y, F_orientation = zones_center(
        F_approaching_zones, group_pos, group_radius, F_limit_points)
    axs[0].plot(F_center_x, F_center_y, 'r.', markersize=5)

    for i, angle in enumerate(F_orientation):
        draw_arrow(F_center_x[i], F_center_y[i], angle, axs[0])

    axs[0].set_xlabel(r'$x$ $[cm]$')
    axs[0].set_ylabel(r'$y$ $[cm]$')
    axs[0].set_title(r'Adaptive Parameters - Perimeter =  %d $cm$' %
                     F_approaching_perimeter)
    axs[0].set_aspect(aspect=1)
    ###########################################################################
    Z_F = np.zeros([N, N])

    for person in persons:
        sigma = None
        sigma_back = None
        Z1 = None
        mu = np.array([person[0], person[1]])
        sigma = params_conversion(
            F_PSPACEX, F_PSPACEY, person[2])

        sigma_back = params_conversion(
            F_PSPACEX / BACK_FACTOR, F_PSPACEY, person[2])

        # The distribution on the variables X, Y packed into pos.
        Z1 = asymmetric_gaussian(
            pos, mu, sigma, person[2], (person[0], person[1]), N, sigma_back)

        # Z matrix only updates the values where Z1 > Z
        cond = None
        cond = Z1 > Z_F
        Z_F[cond] = Z1[cond]

        plot_person(person[0], person[1], person[2], axs[1], plot_kwargs)

    H_approaching_area = plot_group(
        group_pos, group_radius, pspace_radius, ospace_radius, axs[1])

    show_group_space = True
    sigma = None
    if show_group_space:
        Z1 = None
        mu = np.array([group_pos[0], group_pos[1]])

        sigma = params_conversion(ospace_radius, ospace_radius, 0)

        Z1 = A * multivariate_gaussian(pos, mu, sigma)
        Z1 = multivariate_normal(mu, sigma).pdf(pos)
        # Normalization

        A1 = 1 / Z1.max()
        Z1 = A1 * Z1

        # Z matrix only updates the values where Z1 > Z
        cond = None
        cond = Z1 > Z_F
        Z_F[cond] = Z1[cond]

    cs1 = axs[1].contour(X, Y, Z_F, cmap="jet", linewidths=0.8, levels=10)

    H_approaching_filter, H_approaching_zones, H_limit_points = approaching_area_filtering(
        H_approaching_area, cs1.allsegs[LEVEL][0])

    H_approaching_filter, H_approaching_zones = approaching_heuristic(
        group_radius, pspace_radius, group_pos, H_approaching_filter, cs1.allsegs[LEVEL][0], H_approaching_zones)

    H_x_approach = [j[0] for j in H_approaching_filter]
    H_y_approach = [k[1] for k in H_approaching_filter]

    H_approaching_perimeter = (
        len(H_x_approach) * 2 * math.pi * group_radius) / len(F_approaching_area[0])

    axs[1].plot(H_x_approach, H_y_approach, 'c.', markersize=5)

    H_center_x, H_center_y, H_orientation = zones_center(
        H_approaching_zones, group_pos, group_radius, H_limit_points)
    axs[1].plot(H_center_x, H_center_y, 'r.', markersize=5)

    for i, angle in enumerate(H_orientation):
        draw_arrow(H_center_x[i], H_center_y[i], angle, axs[1])

    axs[1].set_xlabel(r'$x$ $[cm]$')
    axs[1].set_ylabel(r'$y$ $[cm]$')
    axs[1].set_title(r'Fixed Parameters - Perimeter =  %d $cm$' %
                     H_approaching_perimeter)

    ########################################################################

    axs[1].set_aspect(aspect=1)
    fig.tight_layout()

    if plot:
        plt.show(block=False)
        print("==================================================")
        input("Hit Enter To Close... ")
        plt.cla()
        plt.clf()
        plt.close()
    return F_approaching_perimeter, H_approaching_perimeter
