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

    def __init__(self):
        """ """
    

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
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.style.use('seaborn-deep')

        fig, ax = plt.subplots(1, 1, tight_layout=True)

        labels = []
            
        labels = ["55 - 45","65 - 55","75 - 65","120 - 110"]
        adaptive_perim = [56284,55445 ,54873 ,55555]
        fixed_perim = [27110,17112 ,10903 ,3250]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, adaptive_perim,
                           width, label='Adaptive Parameters')
        rects2 = ax.bar(x + width / 2, fixed_perim,
                           width, label='Fixed Parameters')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(r'Perimeter $(cm)$')
        ax.set_xlabel(r'Personal Space Initial Dimensions $(cm)$')
        ax.set_title(r'Approaching Perimeter Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)


        plt.grid(True)
        # #plt.savefig('figures/histograms_2.eps', format='eps')
        plt.show(block=False)
        print("==================================================")
        input("Hit Enter To Close... ")
        plt.close()


def main():
    bar = Bar()
    bar.solve()
    




if __name__ == "__main__":
    main()
