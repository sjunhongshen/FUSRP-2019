import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import *


dataPoints = [[2, 3, 10], [3, 9, 2], [6, 8, 5]]  # first is the root
ini_radius = [2, 1, 1]  # with Murray's Law


# initial point is the average of the coordinates
def ini_posit(dataPoints):
    tempX = 0.0
    tempY = 0.0
    tempZ = 0.0

    for i in range(0, len(dataPoints)):
        tempX += dataPoints[i][0]
        tempY += dataPoints[i][1]
        tempZ += dataPoints[i][2]

    return [tempX / len(dataPoints), tempY / len(dataPoints), tempZ / len(dataPoints)]


# this cost function consider only about length
def cost(testMedian, dataPoints, testRadius):
    temp = 0.0
    for i in range(0, len(dataPoints)):
        temp += math.sqrt((testMedian[0]-dataPoints[i][0])**2 +
                          (testMedian[1]-dataPoints[i][1])**2 +
                          (testMedian[2]-dataPoints[i][2])**2) * (testRadius[i] ** 2) + \
                max(0, testRadius[i]-2)**2*6 + max(0, 1-testRadius[i])**2*6
    return temp


# take gradient to decide the change of coordinates of each step
def gradx(testMedian, dataPoints, testRadius):
    testRadius[2] = (testRadius[0] ** 3 - testRadius[1] ** 3) ** (1/3)
    dx = 0.0
    for i in range(0, len(dataPoints)):
        dx += (testMedian[0] - dataPoints[i][0]) * testRadius[i] ** 2 / \
                math.sqrt((testMedian[0] - dataPoints[i][0])**2 +
                          (testMedian[1] - dataPoints[i][1])**2 +
                          (testMedian[2] - dataPoints[i][2])**2)
    return dx


def grady(testMedian, dataPoints, testRadius):
    testRadius[2] = (testRadius[0] ** 3 - testRadius[1] ** 3) ** (1/3)
    dy = 0.0
    for i in range(0, len(dataPoints)):
        dy += (testMedian[1] - dataPoints[i][1]) * testRadius[i] ** 2 / \
                math.sqrt((testMedian[0] - dataPoints[i][0])**2 +
                          (testMedian[1] - dataPoints[i][1])**2 +
                          (testMedian[2] - dataPoints[i][2])**2)
    return dy


def gradz(testMedian, dataPoints, testRadius):
    testRadius[2] = (testRadius[0] ** 3 - testRadius[1] ** 3) ** (1/3)
    dz = 0.0
    for i in range(0, len(dataPoints)):
        dz += (testMedian[2] - dataPoints[i][2]) * testRadius[i] ** 2 / \
                math.sqrt((testMedian[0] - dataPoints[i][0])**2 +
                          (testMedian[1] - dataPoints[i][1])**2 +
                          (testMedian[2] - dataPoints[i][2])**2)
    return dz


# the gradient of the position
def gradient_posit(testMedian, dataPoints, testRadius):
    return array([gradx(testMedian, dataPoints, testRadius),
                  grady(testMedian, dataPoints, testRadius),
                  gradz(testMedian, dataPoints, testRadius)])


# the gradient of the radius
def gradient_radiu(testMedian, dataPoints, testRadius):
    testRadius[2] = (testRadius[0] ** 3 - testRadius[1] ** 3) ** (1/3)
    dr_1 = 0  # the main radius never change
    dr_2 = 2 * testRadius[1] * \
           math.sqrt((testMedian[0] - dataPoints[1][0])**2 +
                     (testMedian[1] - dataPoints[1][1])**2 +
                     (testMedian[2] - dataPoints[1][2])**2) + 12*(2*testRadius[1]-3)
    dr_3 = 2 * testRadius[2] * \
           math.sqrt((testMedian[0] - dataPoints[2][0])**2 +
                     (testMedian[1] - dataPoints[2][1])**2 +
                     (testMedian[2] - dataPoints[2][2])**2) + 12*(2*testRadius[2]-3)
    return array([dr_1, dr_2, dr_3])


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
