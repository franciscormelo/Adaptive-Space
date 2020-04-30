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
from sklearn.preprocessing import normalize
from matplotlib import rc

from scipy.stats import multivariate_normal

# CONSTANTS
# Human Body Dimensions top view in cm
HUMAN_Y = 45
HUMAN_X = 20

BACK_FACTOR = 1.3


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


def plot_group(group_pose, group_radius, ax):
    """Plots the group o space, p space and approaching circle area. """
    # O Space Modeling
    ax.plot(group_pose[0], group_pose[1], 'rx', markersize=8)
    plot_kwargs = {'color': 'r', 'linestyle': '-', 'linewidth': 1}

    ospace_radius = group_radius - HUMAN_X / 2
    plot_ellipse(semimaj=group_radius - HUMAN_X / 2, semimin=ospace_radius, x_cent=group_pose[0],
                 y_cent=group_pose[1], ax=ax, plot_kwargs=plot_kwargs)

    # P Space Modeling
    psapce_radius = group_radius + HUMAN_X / 2
    plot_ellipse(semimaj=group_radius + HUMAN_X / 2, semimin=psapce_radius, x_cent=group_pose[0],
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


def assymetric_gaussian(pos, mu, Sigma, orientation, center, N, Sigma_back):
    """ """
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

# def approachingfiltering_gaussian(Z,X, Y approaching_area):
#     """Filters the approaching area."""
#     # Approaching Area filtering - remove points tha are inside the personal space of a person
#     if idx == 1:
#         approaching_filter = [(x, y) for x, y in zip(
#             approaching_filter[0], approaching_filter[1]) if not personal_space.contains_point([x, y])]
#     else:
#         cx = [j[0] for j in approaching_filter]
#         cy = [k[1] for k in approaching_filter]
#         approaching_filter = [(x, y) for x, y in zip(
#             cx, cy) if not personal_space.contains_point([x, y])]
#     return approaching_filter


def plot_gaussians(persons, group_pos, group_radius, ellipse_param, N=200, show_group_space=True):
    """ Plots surface and contour of 2D Gaussian function given ellipse parameters."""
    A = 1
    x = [item[0] for item in persons]
    y = [item[1] for item in persons]

    xmin = min(x) - 150
    xmax = max(x) + 150
    ymin = min(y) - 150
    ymax = max(y) + 150

    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
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
        Z1 = assymetric_gaussian(
            pos, mu, Sigma, person[2], (person[0], person[1]), N, Sigma_back)

        #Z1 = multivariate_normal(mu, Sigma).pdf(pos)
        #Z = Z1

        # Z matrix only updates maximum values
        cond = Z1 > Z
        Z[cond] = Z1[cond]

        plot_person(person[0], person[1], person[2], ax2, plot_kwargs)

    approaching_area = plot_group(group_pos, group_radius, ax2)
    # # possible approaching positions
    # approaching_x = [j[0] for j in approaching_filter]
    # approaching_y = [k[1] for k in approaching_filter]
    #
    # ax.plot(approaching_x, approaching_y, 'c.', markersize=5)

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

        # Z matrix only updates maximum values -> When personal
        cond = Z1 > Z
        Z[cond] = Z1[cond]

    surf = ax1.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=1,
                            antialiased=False, cmap="jet")

    ax1.set_xlabel(r'$x (cm)$')
    ax1.set_ylabel(r'$y (cm)$')
    ax1.set_zlabel(r'$Cost$')

    CS = ax2.contour(X, Y, Z, cmap="jet", linewidths=0.8, levels=10)
    fig.colorbar(CS)

    ax2.set_xlabel(r'$x (cm)$')
    ax2.set_ylabel(r'$y (cm)$')
    ax2.set_aspect(aspect=1)
    fig.tight_layout()
    plt.show(block=False)
    print("==================================================")
    input("Hit Enter To Close... ")
    surf.remove()
    plt.clf()
    plt.close()
