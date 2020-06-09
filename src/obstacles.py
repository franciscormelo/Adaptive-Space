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





# Gets the coordinates of a windows around the group
xmin = 0
xmax = 10
ymin = 0
ymax = 10

N = 50

X = np.linspace(xmin, xmax, N)
Y = np.linspace(ymin, ymax, N)

X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.zeros(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

Z = np.zeros([N, N])

fig,ax = plt.subplots(1, 1, tight_layout=True)
im = plt.imshow(Z, cmap="jet", extent=[xmin,xmax,ymin,ymax],origin="lower")

pos = []
def onclick(event):
    pos.append([event.xdata,event.ydata])
    i_x = int((event.xdata*N)/xmax) 
    i_y = int((event.ydata * N)/ymax) 
    print(i_x)
    print(i_y)
    Z[i_y, i_x] = 1
    im.set_data(Z)
    im.autoscale()
    fig.canvas.draw()
     
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()