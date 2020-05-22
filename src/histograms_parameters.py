#! /usr/bin/env python3
'''
    File name: histograms_plot.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
    https://malithjayaweera.com/2018/09/add-matplotlib-percentage-ticks-histogram/
'''
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Histogram:
    """Creates Histograms for personal space parameters."""

    def __init__(self, fh):
        """ """
        # Lists Initialization

        # split the file into lines
        lines = fh.read().splitlines()[1:]

        # removes blank lines
        file = list(filter(None, lines))

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

    def histogram_1d(self, axs, n_bins, x, y):
        """ Plots 1d histogram"""
    # We can set the number of bins with the `bins` kwarg
        N, bins, patches = axs[0].hist([x, y], bins='auto', label=['x', 'y'],
                                       alpha=0.7, rwidth=0.85)
        axs[0].set_xlabel(r'Personal Space $x,y$ $(cm)$')
        axs[0].set_ylabel(r'Frequency')
        axs[0].grid(axis='y', alpha=0.75)
        axs[0].legend(loc='upper right')
        axis_y = axs[0].twinx()
        axis_y.hist([x, y], bins='auto', label=['x', 'y'],
                    alpha=0.7, rwidth=0.85)
        axis_y.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(x)))
        axis_y.set_ylabel(r'Percentage', rotation=270, labelpad=20)

    def histogram_2d(self, axs, n_bins, x, y):
        """ """

        cset1 = axs[1].hist2d(x, y, bins=n_bins)
        axs[1].set_xlabel(r'Personal Space $x$ $(cm)$')
        axs[1].set_ylabel(r'Personal Space $y$ $(cm)$')
        axs[1].grid(axis='y', alpha=0.75)

        cbar = plt.colorbar(cset1[3], ax=axs[1])
        cbar.ax.set_ylabel('Frequency', rotation=270, labelpad=10)
        #plt.savefig('figures/histograms_full.eps', format='eps')
        cbar_per = plt.colorbar(cset1[3], ax=axs[1])
        cbar_per.ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=len(x)))
        cbar_per.ax.set_ylabel('Percentage', rotation=270, labelpad=10)

    def solve(self, number_elements):
        """ Plots Histogram of x and y parameters of personal space"""
        x = self.group_info['param_x']
        y = self.group_info['param_y']

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.style.use('seaborn-deep')

        n_bins = 20

        fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

        self.histogram_1d(axs, 20, x, y)
        axs[0].set_title(r'Persons Space Histogram $x,y$ axis - All Groups')
        self.histogram_2d(axs, 20, x, y)
        axs[1].set_title(r'Personal Space 2D Histogram - All Groups')
        print("Number of groups: " + str(len(x)))

        fig2, axs2 = plt.subplots(1, 2, sharey=False, tight_layout=True)
        g_x = []
        g_y = []

        for i in range(len(self.group_info['group_nb'])):

            if self.group_info['group_nb'][i] == number_elements:
                g_x.append(self.group_info['param_x'][i])
                g_y.append(self.group_info['param_y'][i])

        self.histogram_1d(axs2, 20, g_x, g_y)
        axs2[0].set_title(
            r'Persons Space Histogram $x,y$ axis - %d Members' % number_elements)
        self.histogram_2d(axs2, 20, g_x, g_y)
        axs2[1].set_title(
            r'Personal Space 2D Histogram - %d Members' % number_elements)

        print("Number of groups: " + str(len(g_x)))

        #plt.savefig('figures/histograms_2.eps', format='eps')
        plt.show(block=False)
        print("==================================================")
        input("Hit Enter To Close... ")
        plt.close()


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
