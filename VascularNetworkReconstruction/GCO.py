import numpy as np
from VascularNetwork import VascularNetwork
import networkx as nx
import matplotlib.pyplot as plt
#from SimulatedAnnealing import SA

class GCO():
    def __init__(self, root_loc, leaf_locs, r_init, f_init, p_init):
        self.VN = VascularNetwork(root_loc, leaf_locs, r_init, f_init, p_init)
        self.root_loc = root_loc
        self.root_r = r_init
        self.leaf_locs = leaf_locs
        self.max_l = 5
        self.merge_threshold = 0.25
        #self.optimizer = SA()
        self.optimizer = None
        self.max_iter = 100

    def initialize(self):
        locs = self.leaf_locs + [self.root_loc]
        opt_point_loc = self.get_centroid(locs)
        opt_point = self.VN.add_branching_point(opt_point_loc)
        leaf_radii = [self.root_r] + [self.VN.r_0 for n in range(len(self.leaf_locs))]
        for i in range(len(self.leaf_locs) + 1):
            self.VN.add_vessel(opt_point, i, leaf_radii[i])

    def local_init(self):
        for n in self.VN.tree.nodes:
            self.VN.tree.nodes[n]['relaxed'] = False

    def relax(self, node):
        neighbor_locs = [self.VN.tree.nodes[n]['loc'] for n in self.VN.tree.neighbors(node)]
        neighbor_radii = [self.VN.tree[node][n]['radius'] if self.VN.tree.nodes[n]['relaxed'] else None for n in self.VN.tree.neighbors(node)]
        #new_loc, new_radii = self.optimizer.optimize(neighbor_locs, neighbor_radii)
        new_loc = (2, 2)
        new_radii = [1, 1, 1]
        self.VN.move_node(node, new_loc)
        i = 0
        for n in self.VN.tree.neighbors(node):
            self.VN.update_radius_and_flow((n, node), new_radii[i])
            i += 1
        self.VN.tree.nodes[node]['relaxed'] = True

    def merge(self, node):
        neighbor_edge_lengths = [self.VN.tree[node][n]['length'] for n in self.VN.tree.neighbors(node)]
        shortest_edge_idx = np.argmin(neighbor_edge_lengths)
        second_shortest_edge_idx = np.argsort(neighbor_edge_lengths)[1]
        if neighbor_edge_lengths[shortest_edge_idx] / neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:
            self.VN.merge(node, list(self.VN.tree.neighbors(node))[shortest_edge_idx])

    def split(self, node):
        neighbor_num = len(list(self.VN.tree.neighbors(node)))
        neighbor_edge_radii = np.array([self.VN.tree[node][n]['radius'] for n in self.VN.tree.neighbors(node)])
        neighbor_edge_lengths = np.array([self.VN.tree[node][n]['length'] for n in self.VN.tree.neighbors(node)])
        edges_to_split = []
        max_rs = 0
        while True:
            target_idx = None
            for i in range(neighbor_num):
                if i in edges_to_split:
                    continue
                edges_to_split.append(i)
                new_edge_r = np.sum(np.array(neighbor_edge_radii[edges_to_split]) ** self.VN.c) ** (1 / self.VN.c)
                #pull_force = self.optimizer.get_neg_derivative(neighbor_edge_radii[edges_to_split], neighbor_edge_lengths[edges_to_split])
                pull_force = 2
                rupture_strength = pull_force - new_edge_r ** 4
                if rupture_strength > max_rs:
                    max_rs = rupture_strength
                    target_idx = i
                edges_to_split.remove(i)
            if target_idx != None:
                edges_to_split.append(target_idx)
            else:
                break
        chosen_nodes = np.array(list(self.VN.tree.neighbors(node)))[edges_to_split]
        chosen_locs = [self.VN.tree.nodes[n]['loc'] for n in chosen_nodes]
        self.VN.split(node, self.get_centroid(chosen_locs + [self.VN.tree.nodes[node]['loc']]), chosen_nodes)

    def get_centroid(self, node_locs):
        return tuple([sum(x) / len(x) for x in zip(*node_locs)])

    def local_opt(self):
        self.local_init()
        for n in list(self.VN.tree.nodes):
            if n in range(len(self.leaf_locs) + 1) or n not in self.VN.tree.nodes:
                continue
            self.relax(n)
        self.visualize()
        for n in list(self.VN.tree.nodes):
            if n in range(len(self.leaf_locs) + 1) or n not in self.VN.tree.nodes:
                continue
            self.merge(n)
        for n in list(self.VN.tree.nodes):
            if n in range(len(self.leaf_locs) + 1) or n not in self.VN.tree.nodes:
                continue
            self.split(n)

    def GCO_opt(self):
        diff_flag = True
        cur_l = self.max_l
        cur_iter = 0
        self.initialize()
        while l_cur >= 0:
            while diff_flag and cur_iter <= self.max_iter:
                #cost_before = self.optimizer.get_cost(self.VN)
                cost_before = 0
                self.local_opt()
                #cost_after = self.optimizer.get_cost(self.VN)
                cost_after = 1
                diff_flag = (cost_before != cost_after)
                cur_iter += 1
            self.VN.prune(cur_l)
            self.VN.reconnect()
            cur_l -= 1

    def visualize(self):
        locs = nx.get_node_attributes(self.VN.tree,'loc')
        nx.draw(self.VN.tree, locs)
        label1 = nx.get_node_attributes(self.VN.tree, 'HS')
        label2 = nx.get_edge_attributes(self.VN.tree, 'length')
        nx.draw_networkx_labels(self.VN.tree, locs, label1)
        nx.draw_networkx_edge_labels(self.VN.tree, locs, edge_labels=label2)
        plt.show()

if __name__ == '__main__':
    g = GCO((0,0),[(0,3),(3,0)],0.1,1,2)
    g.GCO_opt()
