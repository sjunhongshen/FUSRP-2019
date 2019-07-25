import numpy as np
from VascularNetwork import VascularNetwork
from SimulatedAnnealing import SA

class GCO():
    def __init__(self, root_loc, leaf_locs, inflow, r_init, f_init, p_init):
        self.VN = VascularNetwork(root_loc, leaf_locs, inflow, r_init, f_init, p_init)
        self.root_loc = root_loc
        self.root_r = r_init
        self.leaf_locs = leaf_locs
        self.l_max = 5
        self.merge_threshold = 0.25
        self.optimizer = SA()

    def initialize(self):
        opt_point_loc = get_centroid(self.leaf_locs.append(self.root_loc))
        opt_point = self.VN.add_branching_point(opt_point_loc)
        leaf_radii = [self.r_init]
        leaf_radii_2 = [self.VN.r_0 for n in range(len(self.leaf_locs))]
        leaf_radii.append(leaf_radii_2)
        for i in range(len(self.leaf_locs) + 1):
            self.VN.add_vessel(opt_point, i, leaf_radii[i])

    def relax(self, node):
        neighbor_locs = [self.VN.tree.nodes[n]['loc'] for n in self.VN.tree.neighbors(node)]
        new_loc, new_radii = self.optimizer.optimize(neighbor_locs)
        self.VN.move_node(node, new_loc)
        i = 0
        for n in self.VN.tree.neighbors(node):
            self.VN.update_radius(tuple(n, node), new_radii[i])
            i += 1

    def merge(self, node):
        neighbor_edge_lengths = [self.VN.tree[node][n]['length'] for n in self.VN.tree.neighbors(node)]
        shortest_edge_idx = np.argmin(neighbor_edge_lengths)
        second_shortest_edge_idx = np.argsort(neighbor_edge_lengths)[1]
        if neighbor_edge_lengths[shortest_edge_idx] / neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:
            self.VN.merge(node, list(self.VN.tree.neighbors(node))[shortest_edge_idx])

    def split(self, node):
        neighbor_num = len(list(self.VN.tree.neighbors(node)))
        neighbor_edge_radii = [self.VN.tree[node][n]['radius'] for n in self.VN.tree.neighbors(node)]
        neighbor_edge_lengths = [self.VN.tree[node][n]['length'] for n in self.VN.tree.neighbors(node)]
        edges_to_split = []
        max_rs = 0
        while True:
            target_idx = None
            for i in range(neighbor_num):
                if i in edges_to_split:
                    continue
                edges_to_split.append(i)
                new_edge_r = np.sum(np.array(neighbor_edge_radii[edges_to_split]) ** self.VN.c) ** (1 / self.VN.c)
                pull_force = self.optimizer.get_neg_derivative(neighbor_edge_radii[edges_to_split], neighbor_edge_lengths[edges_to_split])
                rupture_strength = pull_force - new_edge_r ** 4
                if rupture_strength > max_rs:
                    max_rs = rupture_strength
                    target_idx = i
                edges_to_split.remove(i)
            if target_idx != None:
                edges_to_split.append(target_idx)
            else:
                break
        chosen_nodes = list(self.VN.tree.neighbors(node))[edges_to_split]
        chosen_locs = [self.VN.tree.nodes[n]['loc'] for n in chosen_nodes]
        self.VN.split(node, get_centroid(chosen_locs.append(self.VN.tree.nodes[node]['loc'])), chosen_nodes)

    def get_centroid(self, node_locs):
        return tuple([sum(x) / len(x) for x in zip(*node_locs)])

    def local_opt(self):
        for n in list(self.VN.tree.nodes):
            if n in range(len(self.leaf_locs) + 1):
                continue
            self.relax(n)
        for n in list(self.VN.tree.nodes):
            if n in range(len(self.leaf_locs) + 1):
                continue
            self.merge(n)
        for n in list(self.VN.tree.nodes):
            if n in range(len(self.leaf_locs) + 1):
                continue
            self.split(n)

    def GCO_opt(self):
        diff_flag = True
        l_cur = self.l_max
        self.initialize()
        while l_cur >= 0:
            while diff_flag:
                cost_before = self.optimizer.get_cost(self.VN)
                self.local_opt()
                cost_after = self.optimizer.get_cost(self.VN)
                diff_flag = (cost_before != cost_after)
            self.VN.prune(l_cur)
            self.VN.reconnect()
            l_cur -= 1
