#! /usr/bin/env python3
'''
    File name: bar_perimeter.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7

'''
import sys
import matplotlib.pyplot as plt
import numpy as np


class Bar:
    """Creates bar chart for perimeter"""

    def __init__(self, fh):
        """ """
        # Lists Initialization

        # split the file into lines
        lines = fh.read().splitlines()[2:]

        # removes blank lines
        file = list(filter(None, lines))

        # Lists intialization
        self.group_info = {'group_nb': [], 'adaptive_perim': [],
                           'fixed_perim': []}

        for num, string in enumerate(file):

            data = string.split("-")

            self.group_info['group_nb'].append(int(data[0]))

            params = data[1]
            params = params[params.find('(') + 1:params.find(')') - 1]

            params_xy = params.split(",")
            self.group_info['adaptive_perim'].append(
                round(float(params_xy[0]), 0))
            self.group_info['fixed_perim'].append(
                round(float(params_xy[1]), 0))

    def autolabel(self, rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def solve(self):
        """ """
        adaptive_perimeter = self.group_info['adaptive_perim']
        fixed_perimeter = self.group_info['fixed_perim']

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.style.use('seaborn-deep')

        fig, ax = plt.subplots(1, 2, tight_layout=True)

        labels = []
        for i, perimeter in enumerate(adaptive_perimeter):
            labels.append("G" + str((i + 1)))

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax[0].bar(x - width / 2, adaptive_perimeter,
                           width, label='Adaptive Parameters')
        rects2 = ax[0].bar(x + width / 2, fixed_perimeter,
                           width, label='Fixed Parameters')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[0].set_ylabel(r'Perimeter $(cm)$')
        ax[0].set_xlabel(r'Groups')
        ax[0].set_title(r'Approaching Perimeter Comparison')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels)
        ax[0].legend()

        self.autolabel(rects1, ax[0])
        self.autolabel(rects2, ax[0])

        sum_adaptive = sum(adaptive_perimeter)
        sum_fixed = sum(fixed_perimeter)
        labels_sum = [""]
        x = np.arange(len(labels_sum))  # the label locations

        rects1_sum = ax[1].bar(
            x - width / 2, round(sum_adaptive, 0), width, label='Adaptive Parameters')
        rects2_sum = ax[1].bar(
            x + width / 2, round(sum_fixed, 0), width, label='Fixed Parameters')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[1].set_ylabel(r'Sum Perimeter $(cm)$')
        ax[1].set_xlabel(r'Type of Parameters')
        ax[1].set_title(r'Approaching Perimeter Sum Comparison')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels_sum)
        ax[1].legend()
        self.autolabel(rects1_sum, ax[1])
        self.autolabel(rects2_sum, ax[1])

        plt.grid(True)
        # #plt.savefig('figures/histograms_2.eps', format='eps')
        plt.show(block=False)
        print("==================================================")
        input("Hit Enter To Close... ")
        plt.close()


def main():
    if len(sys.argv) == 2:
        file = "data/" + sys.argv[1]

        with open(file) as fh:
            bar = Bar(fh)
            bar.solve()

            fh.close()

    else:
        print("Usage: %s <filename>  nb_elements" % (sys.argv[0]))


if __name__ == "__main__":
    main()
