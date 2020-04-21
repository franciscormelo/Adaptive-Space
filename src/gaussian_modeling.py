# Adapted from: https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma

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
HUMAN_Y = 62.5
HUMAN_X = 37.5


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


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    n = mu.shape[0]  # Dimension
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.

    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def params_conversion(sx, sy, angle):
    """ Converts ellipses parameteres to Covarince matrix based on the orientation."""
    # https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

    R = np.matrix([[math.cos(angle), -math.sin(angle)],
                   [math.sin(angle), math.cos(angle)]])
    S = np.matrix([[sx / 2, 0.], [0., sy / 2]])
    T = R * S
    covariance = T * T.transpose()

    return covariance


def plot_gaussians(persons, group_pos, group_radius, ellipse_param, N=200, show_group_space=True):
    """ Plots surface and contour of 2D Gaussian function given ellipse parameters."""
    A = 1
    x = [item[0] for item in persons]
    y = [item[1] for item in persons]

    xmin = min(x) - 100
    xmax = max(x) + 100
    ymin = min(y) - 100
    ymax = max(y) + 100

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

        # The distribution on the variables X, Y packed into pos.
        Zg = multivariate_gaussian(pos, mu, Sigma)
        A = 1 / Zg.max()
        Z1 = A * Zg
        #Z1 = multivariate_normal(mu, Sigma).pdf(pos)
        #Z = Z1
        Z = Z + Z1

        plot_person(person[0], person[1], person[2], ax2, plot_kwargs)

    show_group_space = False
    if show_group_space:
        Z1 = None
        mu = np.array([group_pos[0], group_pos[1]])

        Sigma = params_conversion(group_radius, group_radius, 0)

        Z1 = A * multivariate_gaussian(pos, mu, Sigma)
        #Z1 = multivariate_normal(mu, Sigma).pdf(pos)
        Z = Z + Z1

        plot_group(group_pos, group_radius, ax2)

    surf = ax1.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0,
                            antialiased=False, cmap="jet")

    ax1.set_xlabel(r'$x (cm)$')
    ax1.set_ylabel(r'$y (cm)$')
    ax1.set_zlabel(r'$Cost$')

    ax2.contour(X, Y, Z, cmap="hsv", linewidths=0.8, levels=9)

    ax2.set_xlabel(r'$x (cm)$')
    ax2.set_ylabel(r'$y (cm)$')
    fig.tight_layout()
    plt.show(block=False)
    print("==================================================")
    input("Hit Enter To Close... ")
    plt.clf()
    plt.close()
