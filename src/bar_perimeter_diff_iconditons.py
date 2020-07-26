#! /usr/bin/env python3
'''
    File name: bar_perimeter_diff_iconditions.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7

'''
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
from matplotlib import rc
font = {'size'   : 10}
matplotlib.rc('font', **font)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# change font
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

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
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        plt.style.use('seaborn-deep')

        fig, ax = plt.subplots(1, 1, tight_layout=True)

        labels = []
            
        labels = ["55 - 45","65 - 55","75 - 65","85 - 75","95 - 85","100 - 90","110 - 100","120 - 110"]
        adaptive_perim = [56743,55446,54873,54177,55224,54980,54495,54588]
        fixed_perim = [26989,17112,10903,6596,3765,2606,1055,297]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, adaptive_perim,
                           width, label='Adaptive Parameters')
        rects2 = ax.bar(x + width / 2, fixed_perim,
                           width, label='Fixed Parameters')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(r'Perimeter Sum $(cm)$')
        ax.set_xlabel(r'Personal Space Initial Dimensions $(cm)$')
        #ax.set_title(r'Approaching Perimeter Sum Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.75)

        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)


        
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
