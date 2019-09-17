"""
Author: Junhong Shen (jhshen@g.ucla.edu)

Description: Simulated Annealing for the specific case of vascular network optimization.
"""

import numpy as np


class SA_Optimizer():
    """ 
    Given a set of nodes, find the optimal location of the branching point and the radius for each vessel 
    connecting the branching point with one of the nodes.
    """

    def __init__(self, locs, radii, init_loc, mode):
        self.dataPoints = locs
        self.num_points = len(self.dataPoints)
        self.testRadii = [radii]
        self.testMedians = [init_loc]
        self.cost_mode = mode
        self.count = 0
        
        # user-defined parameters
        self.T = 1
        self.a = 0.99
        self.w1 = 6
        self.w2 = 1
        self.w3 = 1e3
        self.c = 3
        self.max_try = 5

        self.costs = [self.cost(self.testMedians[self.count], self.testRadii[self.count])]
        self.dim = len(self.dataPoints[0])


    def optimize(self):
        """ 
        Take random moves and find the minimum cost in all the moves.
        """

        while self.T > 1e-5:
            i = 0
            while i < self.max_try:
                self.move()
                cost_old = self.costs[self.count - 1]
                cost_new = self.costs[self.count]
                if cost_new < cost_old:
                    break
                prob = np.exp((cost_old - cost_new) / self.T)
                pa = np.random.random()
                if prob <= pa:
                    del self.testMedians[-1]
                    del self.testRadii[-1]
                    del self.costs[-1]
                    self.count -= 1
                i += 1
            self.T = self.T * self.a
        min_idx = np.argmin(self.costs)
        return self.testMedians[min_idx], self.testRadii[min_idx], self.costs[min_idx]

    def move(self):
        """ 
        Take a random move.
        """

        loc_new = self.testMedians[self.count] + (2 * np.random.rand(1, self.dim)[0] - 1) * 0.05 * self.T * self.get_loc_range()
        rand = np.concatenate((np.array([0.5]), np.random.rand(1, self.num_points - 1)[0]))
        radii_new = self.testRadii[self.count] + (2 * rand - 1) * 0.05 * self.T * self.get_radius_range()
        radii_new[0] = self.get_first_r(radii_new)
        self.count += 1
        self.testMedians.append(loc_new)
        self.testRadii.append(radii_new)
        self.costs.append(self.cost(self.testMedians[self.count], self.testRadii[self.count]))


    def cost(self, testMedian, testRadius):
        """ 
        Calculate cost.
        """

        # material cost only
        if self.cost_mode == 'MC':
            temp = 0.0
            for i in range(self.num_points):
                temp += np.linalg.norm(testMedian - self.dataPoints[i]) * (testRadius[i] ** 2) + \
                    max(0, testRadius[i]-2)**2*self.w1 + max(0, 1-testRadius[i])**2*self.w2
            return temp

        # material cost and power cost
        else:
            temp1 = 0.0
            for i in range(self.num_points):
                temp1 += np.linalg.norm(testMedian - self.dataPoints[i]) * testRadius[i] ** 2
            temp1 *= self.w1
            temp2 = 0.0
            for i in range(1, len(self.dataPoints)):
                temp2 += testRadius[i] ** 4 / np.linalg.norm(testMedian - self.dataPoints[i])
            temp2 = 1 / temp2 + np.linalg.norm(testMedian - self.dataPoints[0]) / testRadius[0] ** 4
            temp2 *= self.w2
            temp3 = 0.0
            for i in range(self.num_points):
                temp3 += self.penalty(testRadius, i)
            temp3 *= self.w3
            return temp1 + temp2 + temp3


    def penalty(self, testRadius, k):
        """ 
        Calculate penalty which limits the radius within a resonable range.
        """

        return max(0, testRadius[k] - 2) ** 2 * self.w1 + max(0, 1 - testRadius[k]) ** 2 * self.w2


    #######################################
    # Other Helper Functions
    #######################################

    def get_loc_range(self):
        # return np.max(self.testMedians) - np.min(self.testMedians)
        return 20

    def get_radius_range(self):
        # return np.max(self.testRadii) - np.min(self.testRadii)
        return 2

    def get_first_r(self, testRadius):
        return ((np.sum(testRadius ** self.c) - testRadius[0] ** self.c)) ** (1 / self.c)
