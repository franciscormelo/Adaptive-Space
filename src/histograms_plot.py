#! /usr/bin/env python3
'''
    File name: histograms_plot.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import PercentFormatter

import numpy as np


class Histogram:
    """Creates Histograms for personal space parameters."""

    def __init__(self, fh):
        # Lists Initialization

        # split the file into lines
        lines = fh.read().splitlines()[1:]

        # removes blank lines
        file = list(filter(None, lines))
        n = len(file)  # Number of groups in the file

        # Lists intialization
        self.group_info = {'group_nb': [], 'param_x': [],
                           'param_y': []}

        for num, string in enumerate(file):

            data = string.split("-")

            self.group_info['group_nb'].append(int(data[0]))

            params = data[1]
            params = params[params.find('(') + 1:params.find(')') - 1]

            params_xy = params.split(",")
            self.group_info['param_x'].append(float(params_xy[0]))
            self.group_info['param_y'].append(float(params_xy[1]))


    def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
        

    def solve(self, number_elements):
        """ Plots Histogram of x and y parameters of personal space"""
        x = self.group_info['param_x']
        y = self.group_info['param_y']
        #fig, ax = plt.subplots(tight_layout=True)
        # hist =
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.style.use('seaborn-deep')

        n_bins = 20

        fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

        # We can set the number of bins with the `bins` kwarg
        N, bins, patches = axs[0].hist([x, y], bins='auto', label=['x', 'y'],
                                       alpha=0.7, rwidth=0.85)
        axs[0].set_xlabel(r'Personal Space $x,y$ $(cm)$')
        axs[0].set_ylabel(r'Frequency')
        axs[0].set_title(r'Persons Space Histogram $x,y$ axis - All Groups')
        axs[0].grid(axis='y', alpha=0.75)
        axs[0].legend(loc='upper right')

        cset1 = axs[1].hist2d(x, y)
        axs[1].set_xlabel(r'Personal Space $x$ $(cm)$')
        axs[1].set_ylabel(r'Personal Space $y$ $(cm)$')
        axs[1].set_title(r'Personal Space 2D Histogram - All Groups')
        axs[1].grid(axis='y', alpha=0.75)
        plt.colorbar(cset1[3], ax=axs[1])
        #plt.savefig('figures/histograms_full.eps', format='eps')
        print("Number of groups: " + str(len(self.group_info['group_nb'])))

        fig2, axs2 = plt.subplots(1, 2, sharey=False, tight_layout=True)
        g_x = []
        g_y = []
        
        for i in range(len(self.group_info['group_nb'])):

            if self.group_info['group_nb'][i] == number_elements:
                g_x.append(self.group_info['param_x'][i])
                g_y.append(self.group_info['param_y'][i])
                

        axs2[0].hist([g_x, g_y], bins='auto', label=['x', 'y'],
                     alpha=0.7, rwidth=0.85)
        axs2[0].set_xlabel(r'Personal Space $x,y$ $(cm)$')
        axs2[0].set_ylabel(r'Frequency')
        axs2[0].set_title(
            r'Persons Space Histogram $x,y$ axis - %d Members' % number_elements)
        axs2[0].grid(axis='y', alpha=0.75)
        axs2[0].legend(loc='upper right')

        cset2 = axs2[1].hist2d(g_x, g_y)
        axs2[1].set_xlabel(r'Personal Space $x$ $(cm)$')
        axs2[1].set_ylabel(r'Personal Space $y$ $(cm)$')
        axs2[1].set_title(
            r'Personal Space 2D Histogram - %d Members' % number_elements)
        axs2[1].grid(axis='y', alpha=0.75)
        plt.colorbar(cset2[3], ax=axs2[1])
        print("Number of groups of " + str(number_elements) +
              " elements: " + str(len(g_x)))

        #plt.savefig('figures/histograms_2.eps', format='eps')
        plt.show(block=False)
        print("==================================================")
        input("Hit Enter To Close... ")
        plt.close()
        pass


def main():
    if len(sys.argv) == 3:
        file = "data/" + sys.argv[1]

        with open(file) as fh:
            hist = Histogram(fh)
            number_elements = int(sys.argv[2])
            hist.solve(number_elements)

            fh.close()

    else:
        print("Usage: %s <filename>  nb_elements" % (sys.argv[0]))


if __name__ == "__main__":
    main()
