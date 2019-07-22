import numpy as np
from VascularNetwork import VascularNetwork

class GCO():
    def __init__(self, root_loc, leaf_locs, inflow, r_init, f_init, p_init):
        self.VN = VascularNetwork(root_loc, leaf_locs, inflow, r_init, f_init, p_init)
        self.l_max = len(leaf_locs)

    def initialize(self):
        self.VN.reconnect()

    def relax(self):
        pass

    def merge(self):
        pass

    def split(self):
        pass

    def get_difference(self):
        pass

    def local_opt(self):
        self.relax()
        self.merge()
        self.split()

    def GCO_opt(self):
        self.initialize()
        l_cur = 0
        while l_cur <= l_max:
            diff_flag = True
            while diff_flag:
                self.local_opt()
                diff_flag = self.get_difference()
            self.VN.prune(l_cur)
            self.VN.reconnect()
            l_cur += 1
