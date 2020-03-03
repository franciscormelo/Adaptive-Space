#! /usr/bin/env python3

'''Version ellipse function'''
import numpy as np
import matplotlib.pyplot as plt
import math
from ellipse import*
import statistics
import sys




def euclideanDistance(x1,y1,x2,y2):
    ''' '''
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def draw_arrow(x, y, angle): # angle in radians
    ''' '''
    r = 10  # or whatever fits you
    plt.arrow(x, y, r*math.cos(angle), r*math.sin(angle),head_length=1,head_width=1, shape = 'full', color = 'blue')

def draw_person_top(x, y, angle, ax):
    ''' '''
    top_y = 62.5/2
    top_x = 37.5/2
    plot_ellipse(semimaj=top_x,semimin=top_y,phi=angle, x_cent=x, y_cent=y, ax = ax)


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
            self.group_pose = (-157.0,-13.0) #o space center point

            #compute group radius given the center of the o-space
            self.group_radius = self.group_radius()


    def group_radius(self):
        ''' '''
        sum_radius = 0
        for i in range(len(self.persons)):
            #average of the distance between the group memebers and the center of the group, o-space radius
            sum_radius = sum_radius + euclideanDistance(self.persons[i][0],self.persons[i][1], self.group_pose[0], self.group_pose[1])
        return sum_radius / len(self.persons)


    def approaching_pose(self):
        ''' '''
        pass

    def solve(self):
        ''' '''
        # (pos_x, pos_y, angles)
        #p_pose = ((1.,2.,math.pi/6),(2.,3.,-math.pi/2),(3.,2.,5*math.pi/6),(2.,1.,math.pi/2)) # 4 ciruclar arragement proximos
        #p_pose = ((1.5,2.,math.pi/6),(2.,2.5,-math.pi/2),(2.5,2.,5*math.pi/6),(2.,1.5,math.pi/2)) # 4 ciruclar arragement proximos
        #p_pose = ((2,2.,math.pi/2),(3.,2.,math.pi/2)) # vis a vis


        group_radius = self.group_radius


        ### compute mean distance between group members
        d_sum = 0
        for i in range(len(self.persons)-1):
            #average of the distance between group members
            d_sum = d_sum +  euclideanDistance(self.persons[i][0],self.persons[i][1], self.persons[i+1][0],self.persons[i+1][1])
        d_mean = d_sum / len(self.persons)


        ## variar a maneira como e calculado tendo em conta o tipo de grupo

        # Scaling factors for personal space
        Sx = d_mean
        Sy = Sx/1.5
        # por um limite!!!!
        # compute personal space por each person

        f, ax = plt.subplots(1)
        idx = 1
        plot_kwargs = {'color':'g','linestyle':'-','linewidth':0.8}
        for person in self.persons:
            draw_arrow(person[0],person[1], person[2]) #orientation arrow angle in radians
            ax.plot(person[0],person[1],'bo', markersize = 8)
            draw_person_top(person[0],person[1], person[2], ax)
            ax.text(person[0]+3,person[1]+3, "$P_" + str(idx) + "$", fontsize=12)


            plot_ellipse(semimaj=Sx,semimin=Sy,phi=person[2], x_cent=person[0], y_cent=person[1], ax = ax, plot_kwargs=plot_kwargs)

            idx = idx + 1
        #O Space
        ax.plot(self.group_pose[0],self.group_pose[1],'rx', markersize = 8)
        plot_kwargs = {'color':'r','linestyle':'-','linewidth':1}
        plot_ellipse(semimaj=group_radius - 20,semimin=group_radius -20, x_cent=self.group_pose[0], y_cent=self.group_pose[1], ax = ax, plot_kwargs=plot_kwargs)

        #p Space
        plot_ellipse(semimaj=group_radius + 20,semimin=group_radius + 20, x_cent=self.group_pose[0], y_cent=self.group_pose[1], ax = ax, plot_kwargs=plot_kwargs)

        # approaching circle area
        plot_kwargs = {'color':'c','linestyle':':','linewidth':2}
        plot_ellipse(semimaj=group_radius ,semimin=group_radius , x_cent=self.group_pose[0], y_cent=self.group_pose[1], ax = ax, plot_kwargs=plot_kwargs)
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')


        plt.savefig('destination_path.eps', format='eps')
        plt.show()

def main():
    if len(sys.argv)>1:
        with open(sys.argv[1]) as fh:
            app = Space_Modeling(fh)
            app.solve()

            fh.close()
    else:
        print("Usage: %s <filename>"%(sys.argv[0]))


if __name__ == "__main__":
    main()
