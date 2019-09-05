import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mayavi import mlab
from VascularNetwork import VascularNetwork
from GD_Optimizer2 import GD_Optimizer
from SA_Optimizer import SA_Optimizer
import SimAnneal
from binvox_rw import read_as_3d_array, dense_to_sparse

class GCO():
    def __init__(self, fixed_locs, leaf_locs, r_init, f_init, p_init, edge_list):
        self.VN = VascularNetwork(fixed_locs, leaf_locs, r_init, f_init, p_init, edge_list)
        self.root_r = r_init
        self.stable_nodes = self.VN.fixed
        self.leaves = self.VN.leaves
        self.max_l = 3
        self.merge_threshold = 0.1
        self.prune_threshold = 8
        self.optimizer = SA_Optimizer
        self.optimizer2 = GD_Optimizer
        self.use_C = True
        self.max_iter_1 = 15
        self.max_iter_2 = np.log2(len(self.leaves)) * 2 + 3
        self.cost_mode = 'PC'
        print("Max iter: %d" % self.max_iter_2)
        print("Num fixed: %d  Num leaf: %d" % (len(self.stable_nodes), len(self.leaves)))

    def initialize(self):
        for node in self.stable_nodes:
            neighbors = np.array(list(self.VN.tree.neighbors(node)))
            leaf_mask = neighbors > len(self.stable_nodes)
            if np.sum(leaf_mask) == 0:
                continue
            neighbor_leaves = neighbors[leaf_mask]
            neighbor_leaf_locs = [self.VN.tree.nodes[n]['loc'] for n in neighbor_leaves]
            locs = np.concatenate((neighbor_leaf_locs, np.array([self.VN.tree.nodes[node]['loc']])))
            opt_point_loc = self.get_centroid(locs)
            opt_point = self.VN.add_branching_point(opt_point_loc)
            # print("initialize opt point %d" % opt_point)
            for i in neighbor_leaves:
                self.VN.add_vessel(opt_point, i, self.VN.r_0)
                self.VN.tree.remove_edge(node, i)
            self.VN.add_vessel(opt_point, node, (len(neighbor_leaves) * self.VN.r_0 ** 3) ** (1 / 3))

    def relax(self, node):
        neighbors = list(self.VN.tree.neighbors(node))
        neighbor_num = len(neighbors)
        if neighbor_num <= 1:
            return
        # neighbor_radii = np.array([self.VN.tree[node][n]['radius'] for n in neighbors])
        # root_idx = np.argmax(neighbor_radii)
        neighbor_order = np.array([self.VN.tree.nodes[n]['level'] for n in neighbors])
        if -1 in neighbor_order:
            root_idx = np.where(neighbor_order == -1)[0][0]
        else:
            root_idx = np.argmax(neighbor_order)
        non_root = neighbors[0]
        neighbors[0] = neighbors[root_idx]
        neighbors[root_idx] = non_root
        neighbor_radii = np.array([self.VN.tree[node][n]['radius'] for n in neighbors])
        neighbor_locs = np.array([self.VN.tree.nodes[n]['loc'] for n in neighbors], dtype=float)
        neighbor_order = np.array([self.VN.tree.nodes[n]['level'] for n in neighbors])
        print("\tnode %d old loc: %s" % (node, self.VN.tree.nodes[node]['loc']))
        print("\tnode %d neighbors: %s" % (node, neighbors))
        print("\tnode %d neighbor radii: %s" % (node, neighbor_radii))
        print("\tnode %d neighbor order: %s" % (node, neighbor_order))
        if self.use_C:
            ret_list = SimAnneal.SA(neighbor_locs[:, 0].copy(), neighbor_locs[:, 1].copy(), neighbor_locs[:, 2].copy(), neighbor_radii)
            new_radii = np.array(ret_list[:neighbor_num])
            new_loc = np.array(ret_list[neighbor_num : neighbor_num + 3])
            cost = ret_list[neighbor_num + 3]
        else:
            local_optimizer = self.optimizer(neighbor_locs, neighbor_radii, self.VN.tree.nodes[node]['loc'], self.cost_mode)
            new_loc, new_radii, cost = local_optimizer.optimize()
        # local_optimizer2 = self.optimizer2(neighbor_locs, neighbor_radii, self.VN.tree.nodes[node]['loc'], self.cost_mode)
        # new_loc2, new_radii2, cost2 = local_optimizer2.optimize()
        print("\tnode %d new loc: %s" % (node, new_loc))
        print("\tnode %d new radii: %s" % (node, new_radii))
        print("\tnode %d cost: %f" % (node, cost))
        self.VN.move_node(node, new_loc)
        i = 0
        for n in neighbors:
            self.VN.update_radius_and_flow((n, node), new_radii[i])
            i += 1

    def merge(self, node):
        neighbor_edge_lengths = [self.VN.tree[node][n]['length'] for n in self.VN.tree.neighbors(node)]
        shortest_edge_idx = np.argmin(neighbor_edge_lengths)
        if list(self.VN.tree.neighbors(node))[shortest_edge_idx] in self.VN.leaves or len(neighbor_edge_lengths) == 1:
            return
        second_shortest_edge_idx = np.argsort(neighbor_edge_lengths)[1]
        if neighbor_edge_lengths[shortest_edge_idx] / neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:
            print("node %d merge" % node)
            if list(self.VN.tree.neighbors(node))[shortest_edge_idx] in self.stable_nodes:
                self.VN.merge(list(self.VN.tree.neighbors(node))[shortest_edge_idx], node)
            else:
                self.VN.merge(node, list(self.VN.tree.neighbors(node))[shortest_edge_idx])

    def split_two_edges(self, node, root_idx):
        neighbors = list(self.VN.tree.neighbors(node))
        neighbor_num = len(neighbors)
        if neighbor_num <= 3:
            return [], 0
        neighbor_edge_radii = np.array([self.VN.tree[node][n]['radius'] for n in neighbors])
        max_rs = 0
        target_edges = []
        for i in range(neighbor_num - 1):
            if i == root_idx: continue
            for j in range(i + 1, neighbor_num):
                if j == root_idx: continue
                edges_to_split = [i, j]
                new_edge_r, _ = self.VN.split_radius(node, np.array(neighbors)[edges_to_split])
                pull_force = np.linalg.norm(self.local_derivative(node, np.array(neighbors)[edges_to_split]))
                rs = self.rupture_strength(pull_force, new_edge_r)
                if rs > max_rs:
                    max_rs = rs
                    target_edges = edges_to_split
        return target_edges, max_rs

    def split(self, node):
        neighbors = list(self.VN.tree.neighbors(node))
        neighbor_num = len(neighbors)
        # neighbor_edge_radii = np.array([self.VN.tree[node][n]['radius'] for n in neighbors])
        # root_idx = np.argmax(neighbor_edge_radii)
        neighbor_order = np.array([self.VN.tree.nodes[n]['level'] for n in neighbors])
        if -1 in neighbor_order:
            root_idx = np.where(neighbor_order == -1)[0][0]
        else:
            root_idx = np.argmax(neighbor_order)
        edges_to_split, max_rs = self.split_two_edges(node, root_idx)
        # print("node %d split start: %s" % (node, edges_to_split))
        while True:
            target_idx = None
            for i in range(neighbor_num):
                if i in edges_to_split or i == root_idx:
                    continue
                edges_to_split.append(i)
                new_edge_r, _ = self.VN.split_radius(node, np.array(neighbors)[edges_to_split])
                pull_force = np.linalg.norm(self.local_derivative(node, np.array(neighbors)[edges_to_split]))
                rs = self.rupture_strength(pull_force, new_edge_r)
                # print("\tedges: %s" % edges_to_split)
                # print("\tnew r: %f pull force: %f rs: %f" % (new_edge_r, pull_force, rs))
                if rs > max_rs:
                    max_rs = rs
                    target_idx = i
                edges_to_split.remove(i)
            if target_idx != None:
                edges_to_split.append(target_idx)
            else:
                break
        if max_rs <= 0 or len(edges_to_split) < 2 or (neighbor_num - len(edges_to_split)) < 2:
            return
        chosen_nodes = np.array(neighbors)[edges_to_split]
        chosen_locs = [self.VN.tree.nodes[n]['loc'] for n in chosen_nodes]
        print("node %d rupture_strength: %f" % (node, max_rs))
        print("node %d split edges: %s" % (node, chosen_nodes))
        self.VN.split(node, self.get_centroid(chosen_locs + [self.VN.tree.nodes[node]['loc']]), chosen_nodes)

    def get_centroid(self, node_locs):
        return tuple([sum(x) / len(x) for x in zip(*node_locs)])

    def local_cost(self, edge):
        node1, node2 = edge
        if self.cost_mode == 'MC':
            return self.VN.tree[node1][node2]['radius'] ** 2 * self.VN.tree[node1][node2]['length']
        else:
            return (self.VN.tree[node1][node2]['radius'] ** 2 + self.VN.tree[node1][node2]['radius'] ** (-4)) * self.VN.tree[node1][node2]['length']

    def global_cost(self):
        cost_list = [self.local_cost(edge) for edge in list(self.VN.tree.edges)]
        return np.sum(cost_list)

    def local_derivative(self, node, neighbors):
        if self.cost_mode == 'MC':
            vecs = [self.VN.tree[node][n]['radius'] ** 2 * ((self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) / self.VN.tree[node][n]['length']) for n in neighbors]
        else:
            vecs = [(self.VN.tree[node][n]['radius'] ** 2 + self.VN.tree[node][n]['radius'] ** (-4)) * ((self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) / self.VN.tree[node][n]['length']) for n in neighbors]
        return np.sum(vecs, axis=0)

    def rupture_strength(self, pull_force, new_edge_r):
        if self.cost_mode == 'MC':
            return pull_force - new_edge_r ** 2
        else:
            return pull_force - new_edge_r ** 2 - new_edge_r ** (-4)

    def local_opt(self, i):
        self.VN.reorder_nodes()
        for n in range(len(self.VN.tree), 0, -1):
            if n in self.stable_nodes or n in self.leaves or n not in self.VN.tree.nodes:
                continue
            self.relax(n)
        if i == self.max_iter_2:
            return
        for n in range(len(self.VN.tree), 0, -1):
            if n in self.stable_nodes or n in self.leaves or n not in self.VN.tree.nodes:
                continue
            self.merge(n)
        self.VN.reorder_nodes()
        for n in range(len(self.VN.tree), 0, -1):
            if n in self.stable_nodes or n in self.leaves or n not in self.VN.tree.nodes:
                continue
            self.split(n)
        print("connected component: %d " % nx.number_connected_components(self.VN.tree))

    def GCO_opt(self):
        cur_l = self.max_l
        count_l = 0
        cur_iter = 0
        self.initialize()
        while cur_iter <= self.max_iter_1:
            print("\nItearation %d" % cur_iter)
            diff_flag = True
            i = 0
            while diff_flag and i <= self.max_iter_2:
                cost_before = self.global_cost()
                print("\nItearation %d[%d]" % (cur_iter, i))
                print("cost before: %f" % cost_before)
                self.local_opt(i)
                cost_after = self.global_cost()
                print("cost after: %f" % cost_after)
                diff_flag = (cost_before != cost_after)
                i += 1
            self.VN.reorder_nodes()
            cur_level = self.VN.get_max_level()
            print("cur_level: %d" % cur_level)
            if cur_level >= self.prune_threshold and cur_iter < self.max_iter_1:
                self.VN.prune(cur_l, 'level')
                count_l += 1
                self.VN.reconnect()
                print("connected component: %d" % nx.number_connected_components(self.VN.tree))
            if count_l == 4:
                cur_l = 1 if cur_l == 1 else cur_l - 1
                count_l = 0
            cur_iter += 1
        self.VN.final_merge()
        print("final connected component: %d" % nx.number_connected_components(self.VN.tree))
        self.save_results()
        self.visualize()
        self.visualize(True, 'HS')

    def visualize(self, with_label=False, mode='level'):
        dim = len(self.VN.tree.nodes[0]['loc'])
        if dim == 2:
            locs = nx.get_node_attributes(self.VN.tree,'loc')
            nx.draw(self.VN.tree, locs, with_labels=False, node_size=20)
            #label1 = nx.get_node_attributes(self.VN.tree, 'HS')
            label2 = nx.get_edge_attributes(self.VN.tree, 'radius')
            #nx.draw_networkx_labels(self.VN.tree, locs, label1)
            #nx.draw_networkx_labels(self.VN.tree, locs)
            #nx.draw_networkx_edge_labels(self.VN.tree, locs, edge_labels=label2)
            plt.show()
        else:
            nodes = dict()
            coords = list()
            connections = list()
            labels = list()
            text_scale = list()
            radii = list()
            for edge in list(self.VN.tree.edges):
                node1, node2 = edge
                if not node1 in nodes:
                    nodes[node1] = len(coords)
                    coords.append(self.VN.tree.nodes[node1]['loc'])
                    if node1 in self.stable_nodes:
                        text_scale.append((3, 3, 3))
                    else:
                        text_scale.append((1, 1, 1))
                    if mode == 'HS' or mode == 'level':
                        labels.append(str(self.VN.tree.nodes[node1][mode]))
                    else:
                        labels.append(str(node1))
                if not node2 in nodes:
                    nodes[node2] = len(coords)
                    coords.append(self.VN.tree.nodes[node2]['loc'])
                    if node2 in self.stable_nodes:
                        text_scale.append((3, 3, 3))
                    else:
                        text_scale.append((1, 1, 1))
                    if mode == 'HS' or mode == 'level':
                        labels.append(str(self.VN.tree.nodes[node2][mode]))
                    else:
                        labels.append(str(node2))
                connections.append([nodes[node1], nodes[node2]])
                radii.append(self.VN.tree[node1][node2]['radius'])
            coords = np.array(coords)
            mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            mlab.clf()
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.5, scale_mode='none', colormap='Blues', resolution=60)   
            mlab.axes() 
        
            # pts.mlab_source.dataset.lines = connections      
            # tube = mlab.pipeline.tube(pts, tube_radius=0.1)
            # mlab.pipeline.surface(tube, color=(0, 0, 0))

            i = 0
            for n in connections:
                n1, n2 = n[0], n[1]
                r = radii[i]
                mlab.plot3d([coords[n1][0], coords[n2][0]], [coords[n1][1], coords[n2][1]], [coords[n1][2], coords[n2][2]], tube_radius = 0.5 * r)
                i += 1

            if not with_label:
                mlab.show()
                return

            for i in range(len(coords)):
                mlab.text3d(coords[i][0], coords[i][1], coords[i][2], labels[i], scale=text_scale[i])      
            mlab.show()

    def save_results(self):
        file_id = 13
        coord_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_coords.npy' % file_id
        connection_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_connections.npy' % file_id
        radius_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_radii.npy' % file_id
        order_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_HS_order.npy' % file_id
        level_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_level_order.npy' % file_id
        nodes = dict()
        coords = list()
        connections = list()
        radii = list()
        order = list()
        l_order = list()
        self.VN.update_final_order('HS')
        self.VN.update_final_order('level')
        self.VN.update_final_radius()
        for edge in list(self.VN.tree.edges):
            node1, node2 = edge
            if not node1 in nodes:
                nodes[node1] = len(coords)
                coords.append(self.VN.tree.nodes[node1]['loc'])
                order.append(self.VN.tree.nodes[node1]['HS'])
                l_order.append(self.VN.tree.nodes[node1]['level'])
            if not node2 in nodes:
                nodes[node2] = len(coords)
                coords.append(self.VN.tree.nodes[node2]['loc'])
                order.append(self.VN.tree.nodes[node2]['HS'])
                l_order.append(self.VN.tree.nodes[node2]['level'])
            connections.append([nodes[node1], nodes[node2]])
            radii.append(abs(self.VN.tree[node1][node2]['radius']))
        np.save(coord_file, coords)
        np.save(connection_file, connections)
        np.save(radius_file, radii)
        print("Save coords, edges and radius.")
        
        np.save(order_file, order)
        np.save(level_file, l_order)   
        print("Save orders.")     

def generate_random_points():
    dim = 3
    num = 5
    coords = np.random.rand(2 * num, dim) * (-10)
    for i in range(num):
        coords[i][1] = -1 * coords[i][0] - 10
        coords[i + num][1] = coords[i + num][0] + 10

    coords2 = np.random.rand(2 * num, dim) * (10)
    for i in range(num):
        coords2[i][1] = coords2[i][0] - 10
        coords2[i + num][1] = -1 * coords2[i + num][0] + 10

    num = 10
    coords3 = np.random.rand(2 * num, dim) * (-10)
    for i in range(num):
        coords3[i][1] = np.random.random_sample() * (-1 * coords3[i][0] - 10)
        coords3[i + num][1] = np.random.random_sample() * (coords3[i + num][0] + 10)

    coords4 = np.random.rand(2 * num, dim) * (10)
    for i in range(num):
        coords4[i][1] = np.random.random_sample() * (coords4[i][0] - 10)
        coords4[i + num][1] = np.random.random_sample() * (-1 * coords4[i + num][0] + 10)

    coords = np.concatenate((coords, coords2))
    coords3 = np.concatenate((coords3, coords4))
    coords = np.concatenate((coords, coords3))

    return coords

def read_coords_from_binvox(fixed_points_file, random_points_file, edge_file, offset=0, f_all=True, half=False):
     with open(fixed_points_file, 'rb') as f, open(random_points_file, 'rb') as r:
        fixed = read_as_3d_array(f)
        random = read_as_3d_array(r)
        f_coords = np.transpose(dense_to_sparse(fixed.data), (1, 0)).copy()
        r_coords = np.transpose(dense_to_sparse(random.data), (1, 0)).copy()
        if half:
            f_idx = []
            for i in range(len(f_coords)):
                if f_coords[i][0] < 256 and np.random.random() < 0.3:
                    f_idx.append(i)
            r_idx = []
            for i in range(len(r_coords)):
                if r_coords[i][0] < 256 and np.random.random() < 0.03:
                    r_idx.append(i)
        elif not f_all:
            f_idx = np.random.choice(f_coords.shape[0], 900, replace=False)
            r_idx = np.random.choice(r_coords.shape[0], 300, replace=False)
        edge_list = np.load(edge_file) + offset
        if f_all:
            return f_coords, r_coords, edge_list
        new_edge_list = []
        for e in edge_list:
            if e[0] in f_idx and e[1] in f_idx:
                new_edge_list.append([np.where(f_idx == e[0])[0][0], np.where(f_idx == e[1])[0][0]])
        return f_coords[f_idx], r_coords[r_idx], new_edge_list

if __name__ == '__main__':
    fixed_points_file = '/Users/kimihirochin/Desktop/mesh/test_1_image_pts.binvox'
    random_points_file = '/Users/kimihirochin/Desktop/mesh/test_1_hemi_surface_0.99.binvox'
    edge_file = '/Users/kimihirochin/Desktop/mesh/test_1_image_edge_list.npy'

    fixed_points_file = '/Users/kimihirochin/Desktop/mesh/test_1_main_1_image_pts.binvox'
    edge_file = '/Users/kimihirochin/Desktop/mesh/test_1_main_1_edge_list.npy'
    fixed_points_file_2 = '/Users/kimihirochin/Desktop/mesh/test_1_main_2_image_pts.binvox'
    edge_file_2 = '/Users/kimihirochin/Desktop/mesh/test_1_main_2_edge_list.npy'
    random_points_file = '/Users/kimihirochin/Desktop/mesh/test_1_hemi_uniform.binvox'
    random_points_file = '/Users/kimihirochin/Desktop/mesh/test_1_p_volume_2_uniform_1.binvox'
    f_coords, r_coords, new_edge_list = read_coords_from_binvox(fixed_points_file, random_points_file, edge_file)
    f_coords_2, _, new_edge_list_2 = read_coords_from_binvox(fixed_points_file_2, random_points_file, edge_file_2, offset=len(f_coords))
    f_coords = np.concatenate((f_coords, f_coords_2))
    new_edge_list = np.concatenate((new_edge_list, new_edge_list_2))

    # f_coords =[[0, -10, 0]]
    # r_coords = generate_random_points()
    # new_edge_list = []

    g = GCO(f_coords, r_coords, 1.8, 10, 2, new_edge_list)
    g.GCO_opt()
