import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class VascularNetwork():
    def __init__(self, fixed, leaves, r_init, f_init, p_init, edge_list):
        self.tree = nx.Graph()
        self.root_r = r_init
        self.r_0 = 0.5
        self.f_0 = f_init / len(leaves)
        self.p_0 = p_init
        self.k = 1
        self.c = 3
        self.mu = 3.6 * 1e-3
        self.alpha = 1
        self.node_count = 0
        self.edge_list = edge_list
        self.fixed = range(len(fixed))
        self.leaves = range(len(fixed), len(fixed) + len(leaves))
        for fixed_loc in fixed:
            self.add_branching_point(fixed_loc, True)
        for leaf_loc in leaves:
            self.add_branching_point(leaf_loc)
        self.initialize_nodes() 

    def initialize_nodes(self, split=True):
        for edge in self.edge_list:
            n1, n2 = edge[0], edge[1]
            self.add_vessel(n1, n2, self.root_r)
        if split:
            count = 0
            level_map = {}
            for i in range(5):
                for edge in list(self.tree.edges):
                    node1, node2 = edge
                    if np.linalg.norm(self.tree.nodes[node1]['loc'] - self.tree.nodes[node2]['loc']) <= 5: continue
                    loc = (self.tree.nodes[node1]['loc'] + self.tree.nodes[node2]['loc']) / 2
                    self.split(node1, loc, [node2])
                    level_map[self.node_count - 1] = len(self.fixed) + count
                    self.tree.nodes[self.node_count - 1]['fixed'] = True
                    count += 1
            print(count)
            for i in self.fixed:
                level_map[i] = i
            for i in self.leaves:
                level_map[i] = i + count
            self.tree = nx.relabel_nodes(self.tree, level_map)
            self.fixed = range(len(self.fixed) + count)
            self.leaves = range(len(self.fixed), len(self.fixed) + len(self.leaves))
            self.edge_list = []
            for edge in list(self.tree.edges):
                node1, node2 = edge
                self.edge_list.append([node1, node2])
        # for node in list(self.tree.nodes):
        #     if self.tree.degree[node] == 0:
        #         print("deg <= 1: %d" % node)
        # return
        for node in list(self.tree.nodes):
            # if self.tree.nodes[node]['fixed'] and self.tree.degree[node] <= 1:
                # dis_list = np.array([np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[n]['loc']) for n in self.fixed])
                # closest_list = np.argsort(dis_list)
                # for n in closest_list:
                #     if n != node and len(list(nx.all_simple_paths(self.tree, source=node, target=n))) == 0:
                #         self.add_vessel(node, n, self.r_0)
                #         e.append([node, n])
                #         print(node)
                #         break
            if not self.tree.nodes[node]['fixed']:
                closest = self.find_nearest_fixed(node)
                self.add_vessel(node, closest, self.r_0)

    def add_branching_point(self, loc, fixed=False, pres=None):
        self.tree.add_node(self.node_count, loc=np.array(loc), pressure=pres, HS=None, level=None, fixed=fixed)
        self.node_count += 1
        return self.node_count - 1

    def add_vessel(self, node, neighbor_node, radius=None, flow=None):
        r = self.r_0 if radius == None else radius
        f = self.f_0 if flow == None else flow
        dis = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[neighbor_node]['loc'])
        self.tree.add_edge(node, neighbor_node, radius=r, flow=f, length=dis)

    def merge(self, node1, node2):
        for n in self.tree.neighbors(node2):
            if node1 != n:
                self.add_vessel(node1, n, self.tree[node2][n]['radius'], self.tree[node2][n]['flow'])
        self.tree.remove_node(node2)

    def split(self, node1, node2_loc, nodes_to_split):
        node2 = self.add_branching_point(node2_loc)
        for n in nodes_to_split:
            self.add_vessel(node2, n, self.tree[node1][n]['radius'], self.tree[node1][n]['flow'])
        r_sum, f_sum = self.split_radius(node1, nodes_to_split)
        for n in nodes_to_split:
            self.tree.remove_edge(node1, n)
        self.add_vessel(node1, node2, r_sum, f_sum)
        print("split and create node %d at loc %s with radius %f" % (node2, node2_loc, r_sum))

    def split_radius(self, node, nodes_to_split):
        if len(nodes_to_split) == 1:
            n = nodes_to_split[0]
            return self.tree[node][n]['radius'], self.tree[node][n]['flow']
        r_sum = 0
        f_sum = 0
        remaining_nodes = list(self.tree.neighbors(node))
        neighbor_edge_radii = np.array([self.tree[node][n]['radius'] for n in remaining_nodes])
        root = remaining_nodes[np.argmax(neighbor_edge_radii)]
        for n in nodes_to_split:
            remaining_nodes.remove(n)
        if root in remaining_nodes:
            remaining_nodes = nodes_to_split
        for n in remaining_nodes:
            r_sum += self.tree[node][n]['radius'] ** self.c
            f_sum += self.tree[node][n]['flow']
        return r_sum ** (1 / self.c), f_sum

    def prune(self, l, mode='level'):
        self.update_order(mode)
        for edge in list(self.tree.edges):
            node1, node2 = edge
            edge1 = [node1, node2]
            edge2 = [node2, node1]
            if edge1 in self.edge_list or edge2 in self.edge_list:
                continue
            # print("node %d level: %d" % (node1, self.tree.nodes[node1][mode]))
            # print("node %d level: %d" % (node2, self.tree.nodes[node2][mode]))
            if self.tree.nodes[node1][mode] == -1 or self.tree.nodes[node2][mode] == -1:
                order = max(self.tree.nodes[node1][mode], self.tree.nodes[node2][mode])
            else:
                order = min(self.tree.nodes[node1][mode], self.tree.nodes[node2][mode])
            if order <= l:
                self.tree.remove_edge(node1, node2)
                print("prune edge (%d, %d)" % (node1, node2))
        for node in list(self.tree.nodes):
            if len(list(self.tree.neighbors(node))) == 0 and node not in self.leaves and node not in self.fixed:
                self.tree.remove_node(node)
                print("prune node %d" % node)

    def reconnect(self):
        for leaf in self.leaves:
            if self.tree.degree[leaf] != 0:
                continue
            nearest_node = self.find_nearest_fixed(leaf)
            min_dis = np.linalg.norm(self.tree.nodes[leaf]['loc'] - self.tree.nodes[nearest_node]['loc'])
            for node in self.tree.nodes:
                dis = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[leaf]['loc'])
                if node not in self.leaves and node not in self.fixed and dis < min_dis:
                    min_dis = dis
                    nearest_node = node
                    # print("leaf %d is closer to %d with distance %f" % (leaf, nearest_node, min_dis))
            self.add_vessel(nearest_node, leaf, self.r_0, self.f_0)
            print("reconnect %d and %d with radius %f" % (leaf, nearest_node, self.r_0))

    def find_nearest_fixed(self, node):
        dis_list = np.array([np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[n]['loc']) for n in self.fixed])
        return np.argsort(dis_list)[0]

    def move_node(self, node, loc_new):
        self.tree.nodes[node]['loc'] = loc_new
        for n in self.tree.neighbors(node):
            self.tree[node][n]['length'] = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[n]['loc'])

    def reorder_nodes(self):
        print("Reordering...")
        self.update_order()
        level_list = []
        node_list = list(self.tree.nodes)
        for node in node_list:
            level_list.append(self.tree.nodes[node]['level'])
        level_idices = np.argsort(-1 * np.array(level_list))
        level_map = {}
        for i in self.fixed:
            level_map[i] = i
        for i in self.leaves:
            level_map[i] = i
        idx = len(self.leaves) + len(self.fixed)
        for i in level_idices:
            if i in self.fixed or i in self.leaves:
                continue
            level_map[node_list[i]] = idx
            idx += 1
        self.tree = nx.relabel_nodes(self.tree, level_map)
        print("Finish.")

    def update_radius_and_flow(self, edge, r_new):
        node1, node2 = edge
        self.tree[node1][node2]['radius'] = r_new
        self.tree[node1][node2]['flow'] = self.k * (r_new ** self.c)

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

    def update_order(self, mode='level'):
        for n in self.tree.nodes:        
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] == 1 and not self.tree.nodes[n]['fixed'] else 0
            if self.tree.nodes[n]['fixed']:
                self.tree.nodes[n][mode] = -1
        count_no_label = self.node_count
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
                if -1 in neighbor_orders and np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue
                if -1 not in neighbor_orders and np.count_nonzero(neighbor_orders == 0) > 1:
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1
            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)
            # print(count_no_label)

    def update_final_order(self, mode='HS'):
        for n in self.tree.nodes:        
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] == 1 else 0
        count_no_label = self.node_count
        cur_order = 1
        cycle_edges = []
        try:
            cycle_edges = list(nx.find_cycle(self.tree))
        except:
            pass
        while len(cycle_edges) != 0:
            print("cycle!")
            self.tree.nodes[cycle_edges[0][0]][mode] = 1
            cycle_edges = []
            try:
                cycle_edges = list(nx.find_cycle(self.tree))
            except:
                pass
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
                if np.count_nonzero(neighbor_orders == 0) > 1:
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1
            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def get_max_level(self):
        return np.max(np.array([self.tree.nodes[n]['level'] for n in self.tree.nodes]))

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
    VN = VascularNetwork((0,0),[(0,4),(4,0)],0.1,3,2)
    VN.add_branching_point((3,3))
    VN.add_branching_point((1, 2))
    VN.add_vessel(0, 1)
    VN.add_vessel(0, 2)
    VN.add_vessel(0, 3)
    VN.add_branching_point((2,1))
    VN.add_vessel(4, 3)
    VN.add_vessel(5, 3)
    VN.reorder_nodes()
    """
    VN.split(3, (1.5,1.5), [4,5])
    VN.split(0, (2,2), [1])
    VN.split(0, (3,1), [2])
    VN.move_node(1, (1, 4))  
    VN.merge(3, 6)  
    VN.prune(1)
    VN.reconnect()
    """
    """
    VN.split(0, (0,1,0), [1])
    VN.split(0, (1,8,9), [2])
    VN.split(0, (1,1,1), [3])
    VN.tree.remove_edge(7,1)
    VN.tree.remove_edge(8,2)
    VN.tree.remove_edge(9,3)
    VN.reconnect()
    plt.subplot(121)
    #nx.draw(VN.tree, with_labels=True, font_weight='bold')
    #plt.show()
    """
    locs = nx.get_node_attributes(VN.tree,'loc')
    nx.draw(VN.tree, locs, with_labels=True)
    label1 = nx.get_node_attributes(VN.tree, 'level')
    label2 = nx.get_edge_attributes(VN.tree, 'length')
    nx.draw_networkx_labels(VN.tree, locs, label1)
    nx.draw_networkx_edge_labels(VN.tree, locs, edge_labels=label2)
    plt.show()






