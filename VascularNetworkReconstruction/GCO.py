import numpy as np
from VascularNetwork import VascularNetwork
import networkx as nx
import matplotlib.pyplot as plt
#from SimulatedAnnealing import SA
from GD_Optimizer import GD_Optimizer

class GCO():
    def __init__(self, root_loc, leaf_locs, r_init, f_init, p_init):
        self.VN = VascularNetwork(root_loc, leaf_locs, r_init, f_init, p_init)
        self.root_loc = root_loc
        self.root_r = r_init
        self.leaf_locs = leaf_locs
        self.max_l = 2
        self.merge_threshold = 0.25
        self.prune_threshold = 1
        self.optimizer = GD_Optimizer
        self.max_iter = 5

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
        self.VN.tree.nodes[0]['relaxed'] = True

    def relax(self, node):
        neighbor_locs = [self.VN.tree.nodes[n]['loc'] for n in self.VN.tree.neighbors(node)]
        print("node %d neighbor locs: %s" % (node, neighbor_locs))
        #neighbor_radii = [self.VN.tree[node][n]['radius'] if self.VN.tree.nodes[n]['relaxed'] else None for n in self.VN.tree.neighbors(node)]
        neighbor_radii = [self.VN.tree[node][n]['radius'] for n in self.VN.tree.neighbors(node)]
        local_optimizer = self.optimizer(neighbor_locs, neighbor_radii, self.VN.tree.nodes[node]['loc'])
        new_loc, new_radii, cost = local_optimizer.optimize()
        print("node %d new loc: %s" % (node, new_loc))
        print("node %d new radii: %s" % (node, new_radii))
        print("node %d cost: %f" % (node, cost))
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
        print(neighbor_edge_lengths[shortest_edge_idx] / neighbor_edge_lengths[second_shortest_edge_idx])
        if neighbor_edge_lengths[shortest_edge_idx] / neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:
            print("node %d merge" % node)
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
                if i in edges_to_split or i == 0:
                    continue
                edges_to_split.append(i)
                new_edge_r = np.sum(np.array(neighbor_edge_radii[edges_to_split]) ** self.VN.c) ** (1 / self.VN.c)
                pull_force = np.linalg.norm(self.local_derivative(node, np.array(list(self.VN.tree.neighbors(node)))[edges_to_split]))
                rs = self.rupture_strength(pull_force, new_edge_r)
                if rs > max_rs:
                    max_rs = rs
                    target_idx = i
                edges_to_split.remove(i)
            if target_idx != None:
                edges_to_split.append(target_idx)
            else:
                break
        if max_rs <= 0:
            return
        chosen_nodes = np.array(list(self.VN.tree.neighbors(node)))[edges_to_split]
        chosen_locs = [self.VN.tree.nodes[n]['loc'] for n in chosen_nodes]
        print("node %d rupture_strength: %f" % (node, max_rs))
        print("node %d split edges: %s" % (node, chosen_nodes))
        self.VN.split(node, self.get_centroid(chosen_locs + [self.VN.tree.nodes[node]['loc']]), chosen_nodes)

    def get_centroid(self, node_locs):
        return tuple([sum(x) / len(x) for x in zip(*node_locs)])

    def local_cost(self, edge):
        node1, node2 = edge
        return self.VN.tree[node1][node2]['radius'] ** 2 * self.VN.tree[node1][node2]['length']

    def global_cost(self):
        cost_list = [self.local_cost(edge) for edge in list(self.VN.tree.edges)]
        return np.sum(cost_list)

    def local_derivative(self, node, neighbors):
        vecs = [self.VN.tree[node][n]['radius'] ** 2 * (self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) for n in neighbors]
        return np.sum(vecs, axis=1)

    def rupture_strength(self, pull_force, new_edge_r):
        return  pull_force - new_edge_r ** 2

    def local_opt(self):
        self.local_init()
        self.VN.reorder_nodes()
        for n in range(len(self.VN.tree)):
            if n in range(len(self.leaf_locs) + 1) or n not in self.VN.tree.nodes:
                continue
            self.relax(n)
            self.visualize()
        for n in range(len(self.VN.tree)):
            if n in range(len(self.leaf_locs) + 1) or n not in self.VN.tree.nodes:
                continue
            self.merge(n)
        for n in range(len(self.VN.tree)):
            if n in range(len(self.leaf_locs) + 1) or n not in self.VN.tree.nodes:
                continue
            self.split(n)

    def GCO_opt(self):
        cur_l = self.max_l
        count_l = 0
        cur_iter = 0
        self.initialize()
        while cur_iter <= self.max_iter:
            print("\nItearation %d" % cur_iter)
            diff_flag = True
            i = 0
            self.visualize()
            while diff_flag and i <= self.max_iter:
                cost_before = self.global_cost()
                print("Itearation %d[%d]" % (cur_iter, i))
                print("cost before: %f" % cost_before)
                self.local_opt()
                cost_after = self.global_cost()
                print("cost after: %f" % cost_after)
                diff_flag = (cost_before != cost_after)
                i += 1
            self.visualize()
            self.VN.reorder_nodes()
            cur_level = self.VN.get_max_level()
            print("cur_level: %d" % cur_level)
            if cur_level >= self.prune_threshold:
                self.VN.prune(cur_l)
                count_l += 1
            #self.visualize()
                self.VN.reconnect()
            if count_l == 3:
                cur_l -= 1
            cur_iter += 1
            #self.visualize()

    def visualize(self):
        locs = nx.get_node_attributes(self.VN.tree,'loc')
        nx.draw(self.VN.tree, locs, with_labels=True)
        #label1 = nx.get_node_attributes(self.VN.tree, 'HS')
        label2 = nx.get_edge_attributes(self.VN.tree, 'radius')
        #nx.draw_networkx_labels(self.VN.tree, locs, label1)
        nx.draw_networkx_labels(self.VN.tree, locs)
        #nx.draw_networkx_edge_labels(self.VN.tree, locs, edge_labels=label2)
        plt.show()

if __name__ == '__main__':
    coords = np.random.rand(10, 2) * 10
    print(coords)
    #g = GCO((0,0),[(0,4),(0,1),(1,3),(3,0),(0.5, 0.25),(5,5)],1,10,2)
    g = GCO((0,0),coords,1,10,2)
    #g.initialize()
    #print(g.local_derivative(3, [1, 2]))
    g.GCO_opt()
