#! /usr/bin/env python3
'''Version using gaussian function'''
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
import statistics
import sys




def euclideanDistance(x1,y1,x2,y2):
    ''' '''
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def draw_arrow(x, y, angle): # angle in radians
    ''' '''
    r = 0.5  # or whatever fits you
    plt.arrow(x, y, r*math.cos(angle), r*math.sin(angle), head_length=0.05,head_width=0.05, shape = 'full', color = 'blue')

class Space_Modeling:

    def __init__(self,fh):
        ''' '''
        # split the file into lines
        lines = fh.read().splitlines()

        # removes blank lines
        file = list(filter(None, lines))

        for string in file:

            group = tuple(string.split(","))
            self.group_nb = len(group)

            self.persons = []
            for person in group:

                pose = tuple(person.split(" "))
                self.persons.append(tuple([float(pose[0][1:]),float(pose[1]), float(pose[2][:-1])]))

            self.persons = tuple(self.persons)


    def group_radius(self):
        ''' '''

        sum_radius = 0
        for i in range(len(self.persons)):
            #average of the distance between the group memebers and the center of the group, o-space radius
            sum_radius = sum_radius + euclideanDistance(self.persons[i][0],self.persons[i][1], self.group_pose[0], self.group_pose[1])
            print(euclideanDistance(self.persons[i][0],self.persons[i][1], self.group_pose[0], self.group_pose[1]))


        return sum_radius / len(self.persons)


    def solve(self):
        ''' '''
        # (pos_x, pos_y, angles)
        #p_pose = ((1.,2.,math.pi/6),(2.,3.,-math.pi/2),(3.,2.,5*math.pi/6),(2.,1.,math.pi/2)) # 4 ciruclar arragement proximos
        #p_pose = ((1.5,2.,math.pi/6),(2.,2.5,-math.pi/2),(2.5,2.,5*math.pi/6),(2.,1.5,math.pi/2)) # 4 ciruclar arragement proximos
        #p_pose = ((2,2.,math.pi/2),(3.,2.,math.pi/2)) # vis a vis


        #compute group radius given the center of the o-space
        self.group_pose = (-157.0,-13.0)
        group_radius = self.group_radius()
        print('########')
        print(group_radius)

        ####################################################
        ### compute mean distance between group members
        d_sum = 0
        for i in range(len(self.persons)-1):
            #average of the distance between group members
            d_sum = d_sum +  euclideanDistance(self.persons[i][0],self.persons[i][1], self.persons[i+1][0],self.persons[i+1][1])
        d_mean = d_sum / len(self.persons)




        x, y = np.mgrid[-300:300:1, -300:300:1]
        position = np.empty(x.shape + (2,))
        position[:, :, 0] = x
        position[:, :, 1] = y


        # https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/
        # http://users.isr.ist.utl.pt/~mir/pub/probability.pdf


        ## variar a maneira como e calculado tendo em conta o tipo de grupo

        # Scaling factors for personal space
        A = 1
        Sx = d_mean/2
        Sy = Sx/2
        # por um limite!!!!
        # compute personal space por each person
        for person in self.persons:

            draw_arrow(person[0],person[1], person[2]) #orientation arrow angle in radians
            plt.plot(person[0],person[1],'ro', markersize = 8)


            R = np.matrix([[math.cos(person[2]),-math.sin(person[2])],[math.sin(person[2]),math.cos(person[2])]])
            S = np.matrix([[Sx , 0],[0 ,Sy]])
            T = R * S


            covariance = T * T.transpose()

            z = A * multivariate_normal([person[0],person[1]], covariance.tolist()).pdf(position)
            plt.contour(x, y, z, 9)


        #group space plot
        plt.plot(self.group_pose[0],self.group_pose[1],'bo', markersize = 8)
        z = A * multivariate_normal([self.group_pose[0],self.group_pose[1]], [[1,0],[0,1]]).pdf(position)
        plt.contour(x, y, z, 9)


        #plt.axis([-2, 8, -2, 8])
        plt.savefig('destination_path.eps', format='eps')

        plt.show()






def main():
    if len(sys.argv)>1:
        file = "data/" + sys.argv[1]
        with open(file) as fh:

            app = Space_Modeling(fh)
            app.solve()

            fh.close()
    else:
        print("Usage: %s <filename>"%(sys.argv[0]))


if __name__ == "__main__":
    main()
