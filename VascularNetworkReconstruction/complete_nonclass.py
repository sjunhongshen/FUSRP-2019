import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import *

dataPoints = [[2, 3, 10], [3, 9, 2], [6, 8, 5]]  # first is the root
ini_radius = [2, 1, 1]  # with Murray's Law


# calculate the last radius
def get_last_r(testRadius):
    radiu_sum = 0.0
    for i in range(1, len(dataPoints)-1):
        radiu_sum += testRadius[i] ** 3
    return (testRadius[0] ** 3 - radiu_sum) ** (1 / 3)


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


# length of the branching to each point
def lengthi(testMedian, dataPoints, l):
    return ((testMedian[0] - dataPoints[l][0]) ** 2 +
            (testMedian[1] - dataPoints[l][1]) ** 2 +
            (testMedian[2] - dataPoints[l][2]) ** 2) ** (1 / 2)


# derivative of length with respect to coordinate
def part_de(testMedian, dataPoints, j):
    dx_par = (testMedian[0] - dataPoints[j][0]) / lengthi(testMedian, dataPoints, j)
    dy_par = (testMedian[1] - dataPoints[j][1]) / lengthi(testMedian, dataPoints, j)
    dz_par = (testMedian[2] - dataPoints[j][2]) / lengthi(testMedian, dataPoints, j)
    return [dx_par, dy_par, dz_par]


# penalty function
def penalty(testRadius, k):
    return max(0, testRadius[k]-2)**2*4 + max(0, 1-testRadius[k])**2*4


# one term in each function
def one_term(testMedian, testRadius, dataPoints):
    temp = 0.0
    for i in range(1, len(dataPoints)):
        temp += testRadius[i] ** 4 / lengthi(testMedian, dataPoints, i)
    return temp ** (-2)


# cost function
def cost(testMedian, testRadius, dataPoints):
    temp1 = 0.0
    for i in range(0, len(dataPoints)):
        temp1 += lengthi(testMedian, dataPoints, i) * testRadius[i]
    temp2 = 0.0
    for i in range(1, len(dataPoints)):
        temp2 += testRadius[i] ** 4 / lengthi(testMedian, dataPoints, i)
    temp2 = 1/temp2 + lengthi(testMedian, dataPoints, 0) / testRadius[0]
    temp3 = 0.0
    for i in range(0, len(dataPoints)):
        temp3 += penalty(testRadius, i)
    return temp1 + temp2 + temp3


# gradient of x coordinate
def gradx(testMedian, dataPoints, testRadius):
    temp4 = 0.0
    for i in range(0, len(dataPoints)):
        temp4 += testRadius[i] * part_de(testMedian, dataPoints, i)[0]
    temp6 = 0.0
    for i in range(1, len(dataPoints)):
        temp6 += testRadius[i] ** 4 / lengthi(testMedian, dataPoints, i) ** 2 * part_de(testMedian, dataPoints, i)[0]
    temp7 = testRadius[0] ** (-4) * part_de(testMedian, dataPoints, 0)[0]
    return temp4 + one_term(testMedian, testRadius, dataPoints) * temp6 + temp7


# gradient of y coordinate
def grady(testMedian, dataPoints, testRadius):
    temp4 = 0.0
    for i in range(0, len(dataPoints)):
        temp4 += testRadius[i] * part_de(testMedian, dataPoints, i)[1]
    temp6 = 0.0
    for i in range(1, len(dataPoints)):
        temp6 += testRadius[i] ** 4 / lengthi(testMedian, dataPoints, i) ** 2 * part_de(testMedian, dataPoints, i)[1]
    temp7 = testRadius[0] ** (-4) * part_de(testMedian, dataPoints, 0)[1]
    return temp4 + one_term(testMedian, testRadius, dataPoints) * temp6 + temp7


# gradient of z coordinate
def gradz(testMedian, dataPoints, testRadius):
    temp4 = 0.0
    for i in range(0, len(dataPoints)):
        temp4 += testRadius[i] * part_de(testMedian, dataPoints, i)[2]
    temp6 = 0.0
    for i in range(1, len(dataPoints)):
        temp6 += testRadius[i] ** 4 / lengthi(testMedian, dataPoints, i) ** 2 * part_de(testMedian, dataPoints, i)[2]
    temp7 = testRadius[0] ** (-4) * part_de(testMedian, dataPoints, 0)[2]
    return temp4 + one_term(testMedian, testRadius, dataPoints) * temp6 + temp7


# the gradient of the position
def gradient_posit(testMedian, dataPoints, testRadius):
    return array([gradx(testMedian, dataPoints, testRadius),
                  grady(testMedian, dataPoints, testRadius),
                  gradz(testMedian, dataPoints, testRadius)])


# the gradient of each radius except for the root
def gradient_radiu(testMedian, dataPoints, testRadius):
    dr = [0]  # the main radius never change
    for i in range(1, len(dataPoints)):
        dr_i = lengthi(testMedian, dataPoints, i) - one_term(testMedian, testRadius, dataPoints) * \
               4/lengthi(testMedian, dataPoints, i) * testRadius[i] ** 3 + 8*(2*testRadius[i]-3)
        dr.append(dr_i)
    return array(dr)


testMedian = ini_posit(dataPoints)  # starting position
testRadius = ini_radius  # the initial radius of three vessels
theta = 0.003  # step length
loop_max = 10000  # maximum loop
epsilon = 1e-7  # the precision

for i in range(loop_max):
    testRadius[len(dataPoints) - 1] = get_last_r(testRadius)
    cost1 = cost(testMedian, testRadius, dataPoints)
    testMediani = testMedian - theta * gradient_posit(testMedian, dataPoints, testRadius)
    testRadiusi = testRadius - theta * gradient_radiu(testMedian, dataPoints, testRadius)
    costi = cost(testMediani, testRadiusi, dataPoints)
    if cost1 - costi > epsilon:
        testMedian = testMediani
        testRadius = testRadiusi
        cost1 = costi
    elif costi - cost1 > epsilon:
        theta = theta * 0.3

print(testMedian, testRadius, cost(testMedian, testRadius, dataPoints))