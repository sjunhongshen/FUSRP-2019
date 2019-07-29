import math
import numpy as np
from scipy import *


dataPoints = [[3,4,1],[3,3,2],[6,8,5]]
# first is the root

#initial point is the average of the coordinates
def ini_posit(dataoPoints):
    tempX = 0.0
    tempY = 0.0
    tempZ = 0.0

    for i in range(0, len(dataPoints)):
        tempX += dataPoints[i][0]
        tempY += dataPoints[i][1]
        tempZ += dataPoints[i][2]

    return [tempX / len(dataPoints), tempY / len(dataPoints), tempZ / len(dataPoints)]

#this cost function consider only about length
def cost(testMedian, dataPoints):
    temp = 0.0
    for i in range(0, len(dataPoints)):
        temp += math.sqrt((testMedian[0]-dataPoints[i][0])**2 +
                          (testMedian[1]-dataPoints[i][1])**2 +
                          (testMedian[2]-dataPoints[i][2])**2)
    return temp

#take gradient to decide the change of coordinates of each step
def gradx(testMedian, dataPoints):
    dx = 0.0
    for i in range(0, len(dataPoints)):
        dx += (testMedian[0] - dataPoints[i][0]) / \
                math.sqrt((testMedian[0] - dataPoints[i][0])**2 +
                          (testMedian[1] - dataPoints[i][1])**2 +
                          (testMedian[2] - dataPoints[i][2])**2)
    return dx


def grady(testMedian, dataPoints):
    dy = 0.0
    for i in range(0, len(dataPoints)):
        dy += (testMedian[1] - dataPoints[i][1]) / \
                math.sqrt((testMedian[0] - dataPoints[i][0])**2 +
                          (testMedian[1] - dataPoints[i][1])**2 +
                          (testMedian[2] - dataPoints[i][2])**2)
    return dy


def gradz(testMedian, dataPoints):
    dz = 0.0
    for i in range(0, len(dataPoints)):
        dz += (testMedian[2] - dataPoints[i][2]) / \
                math.sqrt((testMedian[0] - dataPoints[i][0])**2 +
                          (testMedian[1] - dataPoints[i][1])**2 +
                          (testMedian[2] - dataPoints[i][2])**2)
    return dz


def gradient(testMedian, dataPoints):
    return array([gradx(testMedian, dataPoints),
                  grady(testMedian, dataPoints),
                  gradz(testMedian, dataPoints)])


testMedian = ini_posit(dataPoints) #starting position
theta = 0.003 #step length
loop_max = 10000 #maximum loop
epsilon = 1e-6 #the precision

for i in range(loop_max):
    cost1 = cost(testMedian, dataPoints)
    testMediani = testMedian - theta * gradient(testMedian, dataPoints)
    costi = cost(testMediani, dataPoints)
    if cost1 - costi > epsilon:
        testMedian = testMediani
        cost1 = costi
    elif costi - cost1 > epsilon:
        theta = theta * 0.3

print(testMedian)
