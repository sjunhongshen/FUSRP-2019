"""
Author: Junhong Shen (jhshen@g.ucla.edu), Abdul Hannan Faruqi (abdul.hannan@zhcet.ac.in)

Description: Defines a blood vascular network and some useful operations.
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import binvox_rw
         

class LeafLocOptimizer:
    """ 
    Given a region of interest represented in 3D binary array, generates a minimum set of points that 
    covers the whole volume with region of influence taken into consideration.
    """

    def __init__(self, region, coords, p, sample_num):
        self.region = np.array(region)
        self.dim = min(self.region.shape)
        self.coords = np.array(coords)
        self.inf = np.ones((p, 1), dtype=int)           # defines region of influence for each point
        self.reg = np.zeros((p, 2, 3), dtype=int)       # defines region covered by each point
        self.min_miss = np.count_nonzero(self.region)
        self.min_miss_coords = None

        # user-defined parameters
        self.p = p                      # initial number of points
        self.max_iter = 100             # maximum number of iteration the algorithm runs
        self.sample_num = sample_num    # the number of points that is added in each iteration
    

    def influence(self):
        """
        Find the region covered by each point according to their region of influence.
        """

        for i in range(self.p):
            self.reg[i] = np.array([[self.coords[i] - self.inf[i], self.coords[i] + self.inf[i]]])
            out_of_bound = np.transpose(np.where(~np.logical_and(self.reg[i] >= 0, self.reg[i] < self.dim)))
            for idx in out_of_bound:
                if self.reg[i][tuple(idx)] < 0: self.reg[i][tuple(idx)] = 0
                if self.reg[i][tuple(idx)] >= self.dim: self.reg[i][tuple(idx)] = self.dim - 1


    def update_inf(self):
        """
        Update region of influence for each point by their relative position in the volume.
        """

        threshold = 180
        upper_brain = 20
        lower_brain = 18
        for i in range(self.p):
            inf[i] = upper_brain if coords[i][2] >= threshold else lower_brain
           

    def optimize(self):
        """
        Find the min set of points that cover the whole region.
        """

        cur_iter = 0
        while cur_iter < self.max_iter:
            print("Iteration %d with %d points" % (cur_iter, self.p))

            # find the region not covered by present points and the points outside the 
            # region of interest
            self.update_inf()
            self.influence()
            miss, outside_pts = self.domain_check()
            miss_num = len(miss)
            print("\tmiss num: %d" % miss_num)
            print("\toutside num: %d" % len(outside_pts))
            if miss_num < self.min_miss:
                self.min_miss = miss_num
                self.min_miss_coords = self.coords.copy()

            # move outside points inside
            if miss_num <= len(outside_pts):
                for i in range(miss_num):
                    self.coords[outside_pts[i]] = miss[i]
                for i in range(miss_num, len(outside_pts)):
                    self.coords[outside_pts[i]] = None
                self.min_miss = 0
                self.min_miss_coords = self.coords.copy()
                return

            # check overlap between influence regions of present points
            dic = self.repeat_check()
            repeated = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:int(0.25 * self.p)]
            repeated = [repeated[i][0] for i in range(len(repeated))]
            out_and_rep = outside_pts + repeated

            # move outside points and overlapped / repeated points inside
            if miss_num <= len(out_and_rep):
                for i in range(miss_num):
                    self.coords[out_and_rep[i]] = miss[i]
            else:
                miss_idx = list(range(miss_num))
                moved = []
                for p in out_and_rep:
                    if p in moved: continue
                    k = np.random.randint(len(miss_idx), size=1)[0]
                    self.coords[p] = miss[miss_idx[k]]
                    del miss_idx[k]
                    moved.append(p)
                print("\tmove num: %d" % len(moved))
                sample_coords = []
                for i in range(self.sample_num):
                    sample_coords.append(np.random.randint(self.dim, size=3))
                self.coords = np.concatenate((self.coords, np.reshape(sample_coords, (self.sample_num, 3))))
                self.inf = np.concatenate((self.inf, np.reshape([self.inf[0] for i in range(self.sample_num)], (self.sample_num, 1))))
                self.reg = np.concatenate((self.reg, np.reshape([self.reg[0] for i in range(self.sample_num)], (self.sample_num, 2, 3))))
                self.p += self.sample_num

            cur_iter += 1
        self.remove_outside()


    def remove_outside(self):
        """
        After optimization, remove points that are still outside the region of interest.
        """

        for i in range(len(self.min_miss_coords)):
            if i >= len(self.min_miss_coords): break
            if self.min_miss_coords[i] is None or not self.region[tuple(self.min_miss_coords[i])]:
                self.min_miss_coords = np.delete(self.min_miss_coords, i, axis=0)


    def repeat_check(self):
        """
        Find repeated points, i.e., their influence regions have overlap with other points.
        """

        repeat = {}
        for i in range(self.p):
            overlap = 0
            for j in range(self.p):
                if i == j: continue
                overlap += self.get_overlap(i, j)
            overlap = int(overlap / (self.p - 1))
            if overlap > 0.25 * (2 * self.inf[i] + 1) ** 3:
                repeat[i + 1] = overlap
        return repeat


    def domain_check(self):
        """
        Find uncovered region and outside points.
        """

        miss = self.region.copy()
        outside_pts = []
        for idx in range(self.p):
            prev_miss = np.count_nonzero(miss)
            start = np.array(self.reg[idx][0], dtype=int)
            len_ = np.array(self.reg[idx][1] - self.reg[idx][0], dtype=int)
            grid = [np.array([x, y, z]) for x in range(len_[0]) for y in range(len_[1]) for z in range(len_[2])]
            for i in range(len(grid)):
                x, y, z = tuple(start + grid[i])
                if x >= self.dim or y >= self.dim or z >= self.dim: continue
                miss[x][y][z] = 0
            if np.count_nonzero(miss) == prev_miss:
                outside_pts.append(idx)
        return np.transpose(np.nonzero(miss), (1, 0)), outside_pts


    def get_overlap(self, i, j):
        """
        Find the size of overlap region between the influence region of point i and that of point j.
        """

        count = 0
        start_i = np.array(self.reg[i][0], dtype=int)
        start_j = np.array(self.reg[j][0], dtype=int)
        len_i = np.array(self.reg[i][1] - self.reg[i][0], dtype=int)
        len_j = np.array(self.reg[j][1] - self.reg[j][0], dtype=int)
        grid_i = [np.array([x, y, z]) for x in range(len_i[0]) for y in range(len_i[1]) for z in range(len_i[2])]
        grid_j = [np.array([x, y, z]) for x in range(len_j[0]) for y in range(len_j[1]) for z in range(len_j[2])] + start_j
        for k in range(len(grid_i)):
            if start_i + grid_i[k] in grid_j:
                count += 1
        return count

    

if __name__ == "__main__":
    # np.random.seed(0)
    # p = 10
    # dim = 50
    # inf = 10
    # r = 15

    # x_init = np.random.randint(dim, size=p)
    # y_init = np.random.randint(dim, size=p)
    # z_init = np.random.randint(dim, size=p)
    # coords = np.transpose(np.array([x_init, y_init, z_init]))

    # model = np.zeros((dim, dim, dim), dtype=int)
    # size = model.shape

    # for idx in range(np.prod(size)):
    #     idx_arr = np.unravel_index(idx, size)
    #     x, y, z = idx_arr
    #     x -= dim / 2
    #     y -= dim / 2
    #     z -= dim / 2
    #     if x ** 2 + y ** 2 + z ** 2 <= r ** 2:
    #         model[idx_arr] = 1
    # print(np.count_nonzero(model))
    # optimizer = leafLocOptimizer(model, coords, inf, p)
    # optimizer.optimize()
    # final_coords = optimizer.min_miss_coords
    # print(final_coords)
    # print(len(final_coords))
    # exit()

    np.random.seed(0)
    p = 2000
    dim = 512
    inf = 20
    r = 15
    sample = 10
    volume_file = None
    save_file = None

    with open(volume_file, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        size = model.data.shape

        x_init = np.random.randint(dim, size=p)
        y_init = np.random.randint(dim, size=p)
        z_init = np.random.randint(dim, size=p) 

        coords = np.transpose(np.array([x_init, y_init, z_init]))

        optimizer = leafLocOptimizer(model.data, coords, inf, p, sample)
        optimizer.optimize()
        final_coords = optimizer.min_miss_coords

        out = np.zeros(size, dtype=int)
        for i in range(len(final_coords)):
            out[tuple(final_coords[i])] = 1

        new_f = open(save_file, "xb")
        new_model = binvox_rw.VoxelModel(np.array(out, dtype=int), model.dims, model.translate, model.scale, model.axis_order)
        binvox_rw.write_binvox(new_model, new_f)
        exit()