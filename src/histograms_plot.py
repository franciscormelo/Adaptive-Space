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

class Histogram:
    """Models the personal space, group space and estimates the possibles approaching areas."""

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

            # File Type 1 - Group inidivudals pose and o space center in input file.
            # File Type 2 - Only group individuals pose in input file.

            data = string.split("-")
      
            self.group_info['group_nb'].append(int(data[0]))

            params = data[1]
            params = params[params.find('(')+1:params.find(')') -1]

            params_xy = params.split(",")
            self.group_info['param_x'].append(float(params_xy[0]))
            self.group_info['param_y'].append(float(params_xy[1]))


 

    def solve(self):
        """ Plots Histogram of x and y parameters of personal space"""
        x = self.group_info['param_x']
        y = self.group_info['param_y']
        fig, ax = plt.subplots(tight_layout=True)
        #hist = ax.hist2d(x, y)
        print(y)
        plt.hist(y, bins=10)

        plt.show(block=False)
        print("==================================================")
        input("Hit Enter To Close... ")
        plt.close()
        pass




def main():
    if len(sys.argv) > 1:
        file = "data/" + sys.argv[1]

        with open(file) as fh:
            hist = Histogram(fh)
            hist.solve()

            fh.close()

    else:
        print("Usage: %s <filename>" % (sys.argv[0]))


if __name__ == "__main__":
    main()
