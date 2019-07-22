import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class VascularNetwork():
    def __init__(self, root, leaves, inflow, r_init, f_init, p_init):
        self.tree = nx.Graph()
        self.node_count = 0
        self.add_branching_point(root, p_init)
        self.leaves= range(1, len(leaves) + 1)
        for leaf_loc in leaves:
            self.add_branching_point(leaf_loc)
        self.c = 3
        self.mu = 2.8
        self.alpha = 1
        self.r_0 = r_init
        self.f_0 = f_init
        self.p_0 = p_init

    def add_branching_point(self, loc, pres=None):
        self.tree.add_node(self.node_count, loc=loc, pressure=pres)
        self.node_count += 1

    def add_vessel(self, node, neighbor_node, radius=None, flow=None):
        radius = self.r_0 if radius == None else radius
        flow = self.f_0 if flow == None else flow
        self.tree.add_edge(node, neighbor_node, radius=radius, flow=flow, length=np.linalg.norm(np.array(self.tree.nodes[node]['loc']) - np.array(self.tree.nodes[neighbor_node]['loc'])))

    def merge(self, node1, node2):
        for n in self.tree.neighbors(node2):
            if node2 != n:
                self.add_vessel(node1, n, self.tree[node2][n]['radius'], self.tree[node2][n]['flow'])
        self.tree.remove_node(node2)

    def split(self, node1, node2_loc, nodes_to_split):
        r_sum = 0
        f_sum = 0
        node2 = self.node_count
        self.add_branching_point(node2_loc)
        for n in nodes_to_split:
            r_sum += self.tree[node1][n]['radius'] ** self.c
            f_sum += self.tree[node1][n]['flow']
            self.add_vessel(node2, n, self.tree[node1][n]['radius'], self.tree[node1][n]['flow'])
            self.tree.remove_edge(node1, n)
        self.add_vessel(node1, node2, r_sum ** (1 / self.c), f_sum)

    def prune(self, condition):
        for edge in list(self.tree.edges):
            if condition(edge):
                node1, node2 = edge
                self.tree.remove_edge(node1, node2)
        for node in list(self.tree.nodes):
            if len(list(self.tree.neighbors(node))) == 0 and node not in self.leaves and node != 0:
                self.tree.remove_node(node)

    def reconnect(self):
        for leaf in self.leaves:
            min_length = np.linalg.norm(np.array(self.tree.nodes[0]['loc']) - np.array(self.tree.nodes[leaf]['loc']))
            nearest_node = 0
            for node in self.tree.nodes:
                if node not in self.leaves and np.linalg.norm(np.array(self.tree.nodes[node]['loc']) - np.array(self.tree.nodes[leaf]['loc'])) < min_length:
                    min_length = np.linalg.norm(np.array(self.tree.nodes[node]['loc']) - np.array(self.tree.nodes[leaf]['loc']))
                    nearest_node = node
            if nearest_node == 0:
                self.add_vessel(nearest_node, leaf, self.r_0, self.f_0)
            else:
                self.add_vessel(nearest_node, leaf, self.get_radius_for_leaf(nearest_node), self.get_flow_for_leaf(nearest_node))

    def move_node(self, node, loc_new):
        self.tree.nodes[node]['loc'] = loc_new
        for n in self.tree.neighbors(node):
            self.tree[node1][n]['length'] = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[n]['loc'])

    def update_radius(self, edge, r_new):
        node1, node2 = edge
        self.tree[node1][node2]['radius'] = r_new

    def update_flow(self, edge, f_new):
        node1, node2 = edge
        self.tree[node1][node2]['flow'] = f_new

    def get_radius_for_leaf(self, node):
        radii = np.array([self.tree[n][node]['radius'] for n in self.tree.neighbors(node)])
        max_r = np.max(radii)
        max_r_power = max_r ** self.c
        rest_power = np.sum(radii ** self.c) - max_r_power
        return (max_r_power - rest_power) ** (1 / self.c)

    def get_flow_for_leaf(self, node):
        flows = np.array([self.tree[n][node]['flow'] for n in self.tree.neighbors(node)])
        max_f = np.max(flows)
        rest = np.sum(flows) - max_f
        return (max_f - rest)

    def get_conductance(self, edge):
        node1, node2 = edge
        return self.alpha * np.pi * (self.tree[node1][node2]['radius'] ** 4) / (8 * self.mu * self.tree[node1][node2]['length'])

    def get_pressure_diff(self, edge):
        node1, node2 = edge
        return self.tree[node1][node2]['flow'] / self.get_conductance(edge)

    def get_power_loss(self, edge):
        node1, node2 = edge
        return (self.tree[node1][node2]['flow'] ** 2) * (1 / self.get_conductance(edge))

    def get_max_stress(self, edge):
        node1, node2 = edge
        return (4 * self.mu * self.tree[node1][node2]['flow']) / (np.pi * self.tree[node1][node2]['radius'] ** 3)

if __name__ == '__main__':
    VN = VascularNetwork((0,0,0),[(0,1,0),(1,0,0)],3,0.1,1,2)
    VN.add_branching_point((0,0,1))
    VN.add_branching_point((1, 2, 1))
    VN.add_vessel(0, 1)
    VN.add_vessel(0, 2)
    VN.add_vessel(0, 3)
    VN.add_branching_point((0,1,1))
    VN.add_vessel(4, 3)
    VN.add_vessel(5, 3)
    VN.split(3, (1,1,1), [4])

    def f(x):
        return True if x[0] > 4 or x[1] > 4 else False
    
    VN.prune(f)
    VN.split(0, (0,1,0), [1])
    VN.split(0, (1,8,9), [2])
    VN.split(0, (1,1,1), [3])
    VN.tree.remove_edge(7,1)
    VN.tree.remove_edge(8,2)
    VN.tree.remove_edge(9,3)
    VN.reconnect()
    plt.subplot(121)
    nx.draw(VN.tree, with_labels=True, font_weight='bold')
    plt.show()






