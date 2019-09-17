"""
Author: Yifan Jiang (yifanyifan.jiang@mail.utoronto.ca)

Description: Given a bifurcation, finds the optimal position of the branching point
    and the optimal radii.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import *

class GD_Optimizer():
    def __init__(self, locs, radii, init_loc, mode, theta=0.003, l_max=10000, eps=1e-7, c=3):
        self.dataPoints = locs
        self.num_points = len(self.dataPoints)
        self.testRadius = radii
        self.testMedian = init_loc
        #self.testMedian = (np.random.rand(1, 2) * np.max(self.dataPoints))[0]
        self.theta1 = theta  # step length
        self.theta2 = theta
        self.loop_max = l_max  # maximum loop
        self.epsilon = eps
        self.c = c
        self.w1 = 6
        self.w2 = 6
        self.cost_mode = mode

    # initial point is the average of the coordinates
    def ini_posit(self, dataPoints):
        tempX = 0.0
        tempY = 0.0
        tempZ = 0.0

        for i in range(0, len(dataPoints)):
            tempX += dataPoints[i][0]
            tempY += dataPoints[i][1]
            tempZ += dataPoints[i][2]

        return [tempX / len(dataPoints), tempY / len(dataPoints), tempZ / len(dataPoints)]

    def get_last_r(self, testRadius):
        return (2 * testRadius[0] ** self.c - (np.sum(testRadius ** self.c) - testRadius[self.num_points - 1] ** self.c)) ** (1 / self.c)

    # this cost function consider only about length
    def cost(self, testMedian, dataPoints, testRadius):
        temp = 0.0
        for i in range(0, len(dataPoints)):
            temp += np.linalg.norm(testMedian - dataPoints[i]) * (testRadius[i] ** 2) + \
                max(0, testRadius[i]-2)**2*self.w1 + max(0, 1-testRadius[i])**2*self.w2
        return temp

    # take gradient to decide the change of coordinates of each step
    def grad_axis(self, testMedian, dataPoints, testRadius, axis_dim):
        dx = 0.0
        for i in range(0, len(dataPoints)):
            dx += (testMedian[axis_dim] - dataPoints[i][axis_dim]) * testRadius[i] ** 2 / \
                np.linalg.norm(testMedian - dataPoints[i])
        return dx

    # the gradient of the position
    def gradient_posit(self, testMedian, dataPoints, testRadius):
        return np.array([self.grad_axis(testMedian, dataPoints, testRadius, i) for i in range(len(dataPoints[0]))])

    # the gradient of the radius
    def gradient_radiu(self, testMedian, dataPoints, testRadius):
        dr = [0]  # the main radius never change
        for i in range(1, self.num_points):
            dr_i = 2 * testRadius[i] * \
                np.linalg.norm(testMedian - dataPoints[i]) + 2*self.w1*(2*testRadius[i]-3)
            dr.append(dr_i)
        return np.array(dr)

    def optimize_single(self):
        cost1 = self.cost(self.testMedian, self.dataPoints, self.testRadius)
        for i in range(self.loop_max):
            # print("\t\t\tpos: %s, rad: %s" % (self.gradient_posit(self.testMedian, self.dataPoints, self.testRadius), self.gradient_radiu(self.testMedian, self.dataPoints, self.testRadius)))
            testMediani = self.testMedian - self.theta1 * self.gradient_posit(self.testMedian, self.dataPoints, self.testRadius)
            testRadiusi = self.testRadius - self.theta2 * self.gradient_radiu(self.testMedian, self.dataPoints, self.testRadius)
            if not np.isnan(self.get_last_r(testRadiusi)):
                testRadiusi[self.num_points - 1] = self.get_last_r(testRadiusi)
            # print("\t\t\ttestmedian: %s" % (testMediani))
            # print("\t\t\ttestrad: %s" % (testRadiusi))
            costi = self.cost(testMediani, self.dataPoints, testRadiusi)
            # print("\t\t\tcost1: %f costi: %f cost dif: %f" % (cost1, costi, cost1 - costi))
            if cost1 - costi > self.epsilon:
                self.testMedian = testMediani
                self.testRadius = testRadiusi
                cost1 = costi
            elif costi - cost1 > self.epsilon:
                self.theta1 = self.theta1 * 0.3
                self.theta2 = self.theta2 * 0.3
        return self.testMedian, self.testRadius, self.cost(self.testMedian, self.dataPoints, self.testRadius)

    def optimize(self, mode="multiple"):
        if mode == "single":
            self.optimize_single()
        elif mode == "multiple":
            testMedians = np.random.rand(5, 2) * (np.max(self.dataPoints) - np.min(self.dataPoints)) + np.min(self.dataPoints)
            testMedians = np.concatenate((np.array([self.testMedian]), testMedians))
            tr = self.testRadius
            costs = []
            for tm in testMedians:
                self.testMedian = tm
                self.testRadius = tr
                self.optimize_single()
                costs.append(self.cost(self.testMedian, self.dataPoints, self.testRadius))
            print(testMedians)
            print(costs)
            idx = np.argmin(np.array(costs))
            self.testMedian = testMedians[idx]
            self.testRadius = tr
            self.optimize_single()
        return self.testMedian, self.testRadius, self.cost(self.testMedian, self.dataPoints, self.testRadius)




if __name__ == '__main__':
    testMedian = ini_posit(dataPoints)  # starting position
    testRadius = ini_radius  # the initial radius of three vessels
    theta = 0.003  # step length
    loop_max = 10000  # maximum loop
    epsilon = 1e-7  # the precision

    for i in range(loop_max):
        testRadius[2] = (testRadius[0] ** 3 - testRadius[1] ** 3) ** (1/3)
        cost1 = cost(testMedian, dataPoints, testRadius)
        testMediani = testMedian - theta * gradient_posit(testMedian, dataPoints, testRadius)
        testRadiusi = testRadius - theta * gradient_radiu(testMedian, dataPoints, testRadius)
        costi = cost(testMediani, dataPoints, testRadiusi)
        if cost1 - costi > epsilon:
            testMedian = testMediani
            testRadius = testRadiusi
            cost1 = costi
        elif costi - cost1 > epsilon:
            theta = theta * 0.3

    print(testMedian, testRadius, cost(testMedian, dataPoints, testRadius))


    p1 = np.array([2, 3, 10])
    p2 = np.array([3, 9, 2])
    p3 = np.array([6, 8, 5])

# These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

# the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

# This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    x = np.linspace(-5,10,10)
    y = np.linspace(-5,10,10)

    X, Y = np.meshgrid(x,y)

    Z=(-a/c)*X - (b/c)*Y + (d/c)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z)

    point1 = [2, 3, 10]
    point2 = [3, 9, 2]
    point3 = [6, 8, 5]
    point4 = [5.31806487, 7.41009082, 5.47937876]
    ax.scatter(point1[0], point1[1], point1[2], color='green')
    ax.scatter(point2[0], point2[1], point2[2], color='green')
    ax.scatter(point3[0], point3[1], point3[2], color='green')
    ax.scatter(point4[0], point4[1], point4[2], color='red')

    plt.show()
