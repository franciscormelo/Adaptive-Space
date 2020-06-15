#! /usr/bin/env python3
'''
    File name: obstacles.py
    Author: Francisco Melo
    Mail: francisco.raposo.melo@tecnico.ulisboa.pt
    Date created: X/XX/XXXX
    Date last modified: X/XX/XXXX
    Python Version: 3.7
'''
import matplotlib.pyplot as plt
import numpy as np

import sys
from algorithm import SpaceModeling

class Obstacles(SpaceModeling):
    """ """
    def __init__(self, fh):
        super().__init__(fh)
        
        
    def fill_object(self, points, costmap):
        """ """
        for x in range(points[0][0], points[1][0]):
            for y in range(points[0][1], points[2][1]):
                costmap[y,x] = 2
        
        return costmap
    
        
    def plt_obstacles(self,N, persons_costmap, map_limits):
        """ """
        
        # Gets the coordinates of a windows around the group
        xmin = 0
        xmax = map_limits[1] + abs(map_limits[0])
        ymin = 0
        ymax = map_limits[3] + abs(map_limits[2])


        X = np.linspace(xmin, xmax, N)
        Y = np.linspace(ymin, ymax, N)

        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.zeros(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        obs_costmap = np.zeros([N, N])
        
        costmap = persons_costmap
        cond = costmap < obs_costmap
        costmap[cond] = obs_costmap[cond]

        fig,ax = plt.subplots(1, 1, tight_layout=True)

        im = plt.imshow(costmap, cmap="jet", extent=[xmin,xmax,ymin,ymax],origin="lower")
        plt.colorbar()
       
        points = []
        
        def onclick(event):
            nonlocal points, obs_costmap
            
            i_x = int((event.xdata*N)/xmax) 
            i_y = int((event.ydata * N)/ymax) 
            points.append([i_x,i_y])
            obs_costmap[i_y, i_x] = 2
            
            if len(points) == 3:
                obs_costmap = self.fill_object(points, obs_costmap)
                points = []
            cond = costmap < obs_costmap
            costmap[cond] = obs_costmap[cond]
            im.set_data(costmap)
            im.autoscale()
            fig.canvas.draw()
        
            
            
        fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()
        fig2,ax2 = plt.subplots(1, 1, tight_layout=True)
        cs = ax2.contour(X, Y, costmap, cmap="jet", linewidths=0.8, levels=10)
        
        fig2.colorbar(cs)

        plt.show()
        return costmap

def main():
    if len(sys.argv) > 1:
        file = "data/" + sys.argv[1]
        
        with open(file) as fh:
            app = Obstacles(fh)
            approaching_poses, persons_costmap, map_limits = app.solve()
            
            N = len(persons_costmap)
            costmap = app.plt_obstacles(N, persons_costmap, map_limits)
            
            fh.close()
            
            fig,ax = plt.subplots(1, 1, tight_layout=True)
            im = plt.imshow(costmap, cmap="jet", origin="lower", extent=map_limits)
            plt.colorbar()
            plt.show()
            
    else:
        print("Usage: %s <filename>" % (sys.argv[0]))
        
        
        
if __name__== "__main__":
    main()
        
    