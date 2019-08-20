import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

def analyze_strahler(order_file):
    order = np.load(order_file)
    print(order)
	N = np.max(order)
    for i in range(-1, N + 1):
        print("%d: %d" % (i, np.count_nonzero(order == i)))

def reconstruct_model(coord_file, connection_file, radius_file, order_file):
    coords = np.load(coord_file)
    connections = np.load(connection_file)
    labels = np.array(range(len(coords)))
    labels = np.load(order_file)
    radii = np.load(radius_file)
    for i in range(len(radii)):
        if radii[i] == 0.8:
            radii[i] = 0.5

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.clf()
    pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.5, scale_mode='none', colormap='Blues', resolution=60)   
    mlab.axes() 
    
    i = 0
    for n in connections:
        n1, n2 = n[0], n[1]
        r = radii[i]
        mlab.plot3d([coords[n1][0], coords[n2][0]], [coords[n1][1], coords[n2][1]], [coords[n1][2], coords[n2][2]], tube_radius = 0.5 * r)
        i += 1
    for i in range(len(coords)):
        mlab.text3d(coords[i][0], coords[i][1], coords[i][2], labels[i], scale=(1, 1, 1))      
    mlab.show()

if __name__ == '__main__':
	coord_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_2_coords.npy'
    connection_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_2_connections.npy'
    radius_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_2_radii.npy'
    order_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_2_order.npy'
    n_order_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_2_neighbor_order.npy'