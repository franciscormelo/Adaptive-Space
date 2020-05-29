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
    # def average_perimeter(self):
    #     """ """
    #     average_perim = [] * max(self.group_info['group_nb'])
    #     for group_nb in self.group_info['group_nb']:
    #         average_perim[group_nb -1 ].append()
            

    def sum_perimeter(self):
        """ Return the sum of perimeter of all groups and grouped by number o members"""
        total_perim_fixed = 0
        total_perim_adaptive = 0

        # Maximum number of members in a group
        max_members = max(self.group_info['group_nb'])
        gperim_adap = [0] * max_members
        gperim_fixed = [0] * max_members
        total_perim_fixed = sum(self.group_info['fixed_perim'])
        total_perim_adaptive = sum(self.group_info['adaptive_perim'])

        for idx,group_nb in enumerate(self.group_info['group_nb']):
            print(idx)
            gperim_adap[group_nb -
                        1] += self.group_info['adaptive_perim'][idx]
            gperim_fixed[group_nb -
                         1] += self.group_info['fixed_perim'][idx]

        return gperim_adap, gperim_fixed, total_perim_fixed, total_perim_adaptive

    def solve(self):
        """ """
        adaptive_perimeter = self.group_info['adaptive_perim']
        fixed_perimeter = self.group_info['fixed_perim']

        gperim_adap, gperim_fixed, total_perim_fixed, total_perim_adaptive = self.sum_perimeter()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.style.use('seaborn-deep')

        fig, ax = plt.subplots(1, 2, tight_layout=True)

        labels = []
        nb_labels = range(max(self.group_info['group_nb']))
        for i in nb_labels:
            labels.append(str((i + 1)))

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax[0].bar(x - width / 2, gperim_adap,
                           width, label='Adaptive Parameters')
        rects2 = ax[0].bar(x + width / 2, gperim_fixed,
                           width, label='Fixed Parameters')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[0].set_ylabel(r'Perimeter $(cm)$')
        ax[0].set_xlabel(r'Number of members')
        ax[0].set_title(r'Approaching Perimeter Comparison')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels)
        ax[0].legend()

        # self.autolabel(rects1, ax[0])
        # self.autolabel(rects2, ax[0])

        labels_sum = [""]
        x = np.arange(len(labels_sum))  # the label locations

        rects1_sum = ax[1].bar(
            x - width / 2, round(total_perim_adaptive, 0), width, label='Adaptive Parameters')
        rects2_sum = ax[1].bar(
            x + width / 2, round(total_perim_fixed, 0), width, label='Fixed Parameters')

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
