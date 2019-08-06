import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import *

class GD_Optimizer():
    def __init__(self, locs, radii, init_loc, theta=0.003, l_max=10000, eps=1e-7, c=3):
        self.dataPoints = locs
        self.num_points = len(self.dataPoints)
        self.testRadius = radii
        self.testMedian = init_loc
        self.theta = theta  # step length
        self.loop_max = l_max  # maximum loop
        self.epsilon = eps
        self.c = c
        self.w1 = 4
        self.w2 = 4

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

    # calculate the last radius
    def get_last_r(self, testRadius):
        radiu_sum = 0.0
        for i in range(1, len(dataPoints) - 1):
            radiu_sum += testRadius[i] ** self.c
        return (testRadius[0] ** self.c - radiu_sum) ** (1 / self.c)

    # length of the branching to each point
    def lengthi(self, testMedian, dataPoints, l):
        return np.linalg.norm(testMedian - dataPoints[l])

    # derivative of length with respect to coordinate
    def part_de(self, testMedian, dataPoints, j):
        par = []
        for i in range(len(dataPoints[0])):
            dx_par = (testMedian[i] - dataPoints[j][i]) / lengthi(testMedian, dataPoints, j)
            par.append(dx_par)
        return par

    # penalty function
    def penalty(self, testRadius, k):
        return max(0, testRadius[k] - 2) ** 2 * self.w1 + max(0, 1 - testRadius[k]) ** 2 * self.w2

    # one term in each function
    def one_term(self, testMedian, testRadius, dataPoints):
        temp = 0.0
        for i in range(1, len(dataPoints)):
            temp += testRadius[i] ** 4 / lengthi(self, testMedian, dataPoints, i)
        return temp ** (-2)

    # cost function
    def cost(self, testMedian, testRadius, dataPoints):
        temp1 = 0.0
        for i in range(0, len(dataPoints)):
            temp1 += lengthi(self, testMedian, dataPoints, i) * testRadius[i]
        temp2 = 0.0
        for i in range(1, len(dataPoints)):
            temp2 += testRadius[i] ** 4 / lengthi(self, testMedian, dataPoints, i)
        temp2 = 1 / temp2 + lengthi(self, testMedian, dataPoints, 0) / testRadius[0]
        temp3 = 0.0
        for i in range(0, len(dataPoints)):
            temp3 += penalty(self, testRadius, i)
        return temp1 + temp2 + temp3

    # gradient of x coordinate
    def gradx(self, testMedian, dataPoints, testRadius, axis):
        temp4 = 0.0
        for i in range(0, len(dataPoints)):
            temp4 += testRadius[i] * part_de(self, testMedian, dataPoints, i)[axis]
        temp6 = 0.0
        for i in range(1, len(dataPoints)):
            temp6 += testRadius[i] ** 4 / lengthi(self, testMedian, dataPoints, i) ** 2 * \
                     part_de(self, testMedian, dataPoints, i)[axis]
        temp7 = testRadius[0] ** (-4) * part_de(self, testMedian, dataPoints, 0)[axis]
        return temp4 + one_term(self, testMedian, testRadius, dataPoints) * temp6 + temp7

    # the gradient of the position
    def gradient_posit(self, testMedian, dataPoints, testRadius):
        return np.array([gradx(self, testMedian, dataPoints, testRadius, i) for i in range(len(dataPoints[0]))])

    # the gradient of each radius except for the root
    def gradient_radiu(self, testMedian, dataPoints, testRadius):
        dr = [0]  # the main radius never change
        for i in range(1, len(dataPoints)):
            dr_i = lengthi(self, testMedian, dataPoints, i) - one_term(self, testMedian, testRadius, dataPoints) * \
                   4 / lengthi(self, testMedian, dataPoints, i) * testRadius[i] ** 3 + 8 * (2 * testRadius[i] - 3)
            dr.append(dr_i)
        return array(dr)

    def optimize(self):
        for i in range(self.loop_max):
            cost1 = self.cost(self.testMedian, self.dataPoints, self.testRadius)
            testMediani = self.testMedian - self.theta * self.gradient_posit(self.testMedian, self.dataPoints, self.testRadius)
            testRadiusi = self.testRadius - self.theta * self.gradient_radiu(self.testMedian, self.dataPoints, self.testRadius)
            testRadiusi[self.num_points - 1] = self.get_last_r(testRadiusi)
            costi = self.cost(testMediani, self.dataPoints, testRadiusi)
            if cost1 - costi > self.epsilon:
                self.testMedian = testMediani
                self.testRadius = testRadiusi
                cost1 = costi
            elif costi - cost1 > self.epsilon:
                self.theta = self.theta * 0.3
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
