import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import seaborn as sns
import pandas as pd

class GCOAnalyzer():
    def __init__(self, coord_file, connection_file, radius_file, HS_file, level_file):
        self.coords = np.load(coord_file)
        self.connections = np.load(connection_file)
        self.radii = np.load(radius_file)
        self.node_HS = np.array(np.load(HS_file), dtype=int)
        self.node_level = np.array(np.load(level_file), dtype=int)
        self.HS = []
        self.levels = []
        self.lengths = []
        for n in self.connections:
            n1, n2 = n[0], n[1]
            self.HS.append(min(self.node_HS[n1], self.node_HS[n2]))
            self.levels.append(min(self.node_level[n1], self.node_level[n2]))
            self.lengths.append(np.linalg.norm(self.coords[n1] - self.coords[n2]))
        self.HS = np.array(self.HS, dtype=int)
        self.levels = np.array(self.levels, dtype=int)
        self.lengths = np.array(self.lengths)
        self.max_HS = np.max(self.HS)

    def analyze_strahler(self):
        mean_r = []
        mean_l = []
        for i in range(self.max_HS + 1):
            idices = np.nonzero(self.HS == i)
            count = np.count_nonzero(self.HS == i)
            rad = [self.radii[idx] for idx in idices]
            length = [self.lengths[idx] for idx in idices]
            r_avg = np.mean(rad)
            l_avg = np.mean(length)
            mean_r.append(r_avg)
            mean_l.append(l_avg)
            print("%d: %d, %f, %f" % (i, count, r_avg, l_avg))
        return mean_r[1:], mean_l[1:]

    def reconstruct_model(self):
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        mlab.clf()
        pts = mlab.points3d(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], scale_factor=0.5, scale_mode='none', colormap='Blues', resolution=60)   
        mlab.axes() 
        
        i = 0
        for n in self.connections:
            n1, n2 = n[0], n[1]
            r = self.radii[i]
            if self.HS[i] <= 2: 
                i += 1
                continue
            mlab.plot3d([self.coords[n1][0], self.coords[n2][0]], [self.coords[n1][1], self.coords[n2][1]], [self.coords[n1][2], self.coords[n2][2]], tube_radius = 1.2)
            i += 1
        # for i in range(len(coords)):
        #     mlab.text3d(coords[i][0], coords[i][1], coords[i][2], labels[i], scale=(1, 1, 1))      
        mlab.show()

    def histogram(self):
        data = self.radii
        color = 'green'
        xlabel = "Vessel Radius"
        ylabel = "Frequency"
        x_range = range(0, 3)
        title = 'Average Vessel Radius'

        sns.set()
        plt.hist([data], color=[color])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x_range)
        plt.title(title)
        plt.show()
        sns.distplot([data], norm_hist=True)

        # branch length
        # strahler order

    def scatterplot(self):
        mean_r, mean_l = self.analyze_strahler()
        raw_data = {'x': range(1, self.max_HS), 'y': mean_r}
        df = pd.DataFrame(raw_data, index=range(1, self.max_HS))
        title = 'Radius v.s. Strahler Order'
        ylabel = 'Mean Radius'

        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("ticks")
        sns.lmplot('x', 'y', data=df, fit_reg=True, scatter_kws={"marker": "D", "s": 100})
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)


if __name__ == '__main__':
    coord_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_4_coords.npy'
    connection_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_4_connections.npy'
    radius_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_4_radii.npy'
    HS_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_4_HS_order.npy'
    level_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_4_level_order.npy'
    analyzer = GCOAnalyzer(coord_file, connection_file, radius_file, HS_file, level_file)
    analyzer.analyze_strahler()
    analyzer.reconstruct_model()