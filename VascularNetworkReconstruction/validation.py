import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import seaborn as sns
import pandas as pd
import networkx as nx

class GCOAnalyzer():
    def __init__(self, coord_file, connection_file, radius_file, HS_file, level_file, file_id):
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
        self.file_id = file_id

    def analyze_strahler(self):
        mean_r = []
        mean_l = []
        for i in range(self.max_HS + 1):
            idices = np.nonzero(self.HS == i)[0]
            count = np.count_nonzero(self.HS == i)
            rad = [self.radii[idx] for idx in idices]
            length = [self.lengths[idx] for idx in idices]
            r_avg = np.mean(rad)
            l_avg = np.mean(length)
            mean_r.append(r_avg)
            mean_l.append(l_avg)
            print("%d: %d, %f, %f" % (i, count, r_avg, l_avg))
        return mean_r[1:], mean_l[1:]

    def mean_branch_length(self):
        mean_l = []
        for i in range(1, self.max_HS + 1):
            idices = np.nonzero(self.HS == i)[0]
            HS_graph = nx.Graph()
            ls = []
            for idx in idices:
                HS_graph.add_edge(self.connections[idx][0], self.connections[idx][1], length=self.lengths[idx])
            for c in nx.connected_components(HS_graph):
                sg = HS_graph.subgraph(c)
                els = [HS_graph[e[0]][e[1]]['length'] for e in sg.edges]
                ls.append(np.sum(els))
            mean_l.append(np.mean(ls))
        return mean_l

    def reconstruct_model(self):
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        mlab.clf()
        pts = mlab.points3d(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], scale_factor=0.5, scale_mode='none', colormap='Blues', resolution=60)   
        mlab.axes() 
        
        i = 0
        for n in self.connections:
            n1, n2 = n[0], n[1]
            r = self.radii[i]
            if self.HS[i] <= 3: 
                i += 1
                continue
            mlab.plot3d([self.coords[n1][0], self.coords[n2][0]], [self.coords[n1][1], self.coords[n2][1]], [self.coords[n1][2], self.coords[n2][2]], tube_radius = 1.2)
            i += 1
        # for i in range(len(coords)):
        #     mlab.text3d(coords[i][0], coords[i][1], coords[i][2], labels[i], scale=(1, 1, 1))      
        mlab.show()

    def histogram(self, content='Strahler', save=False):
        sns.set()
        bins = None
        if content == 'Radius':
            data = self.radii 
            xlabel = "Vessel Radius (mium)"
            ylabel = "Relative Frequency"
            x_range = range(0, 5, 1)
            title = 'Vessel Radius'
            kde = True
        elif content == 'Diameter':
            data = self.radii * 2
            xlabel = r'Vessel Diameter ($mm$)'
            ylabel = "Relative Frequency"
            x_range = np.array(range(0, 40, 5)) / 10
            plt.xlim((0, 4))
            title = 'Vessel Diameter'
            kde = True
        elif content == 'Log Diameter':
            from scipy.stats import norm
            data = 1 / ((self.radii * 20) ** 1/2)
            xlabel = r'Inverse of Square Root Diameter $(mm)$'
            ylabel = "Relative Frequency"
            x_range = np.array(list(range(0, 6))) / 10
            bins = np.array(list(range(0, 60))) / 100
            title = 'Inverse of Square Root Diameter'
            plt.xlim((0, 0.6))
            kde = False
            data2 = data[(data <= 0.5)]
            mu, std = norm.fit(data2)
            x = np.linspace(0, 0.6, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
        elif content == 'Length':
            data = self.lengths
            xlabel = r'Branch Length ($mm$)'
            ylabel = "Relative Frequency"
            x_range = range(0, 45, 5)
            plt.xlim((0, 45))
            title = 'Branch Length'
            kde = True
        elif content == 'Log Length':
            from scipy.stats import norm
            data = np.log(self.lengths)
            xlabel = r'Log Length $(mm)$'
            ylabel = "Relative Frequency"
            x_range = np.array(list(range(0, 6)))
            title = 'Log Length'
            plt.xlim((-1, 6))
            kde = False
            mu, std = norm.fit(data)
            x = np.linspace(-1, 6, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
        elif content == 'Strahler':
            print(self.max_HS)
            data = self.HS
            xlabel = r'Strahler Order'
            ylabel = "Relative Frequency"
            x_range = range(1, self.max_HS + 1)
            plt.xlim((1, self.max_HS + 1))
            title = 'Strahler Order'
            kde = False

        
        # plt.hist([data], color=[color])
        sns.distplot([data], bins=bins, kde=kde, kde_kws={'bw':0.25}, norm_hist = True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xlim((0, 4))
        plt.xticks(x_range)
        plt.title(title)

        if save:
            path = '/Users/kimihirochin/Desktop/mesh/plots/test_1_hist_%d_%s.png' % (self.file_id, title)
            plt.savefig(path)
        else:
            plt.show()

    def scatterplot(self, content="Length", save=False):
        if content == "Diameter":
            mean_r, mean_l = self.analyze_strahler()
            raw_data = {'x': range(1, self.max_HS + 1), 'y': np.array(mean_r) * 2}
            df = pd.DataFrame(raw_data, index=range(1, self.max_HS + 1))
            title = 'Diameter v.s. Strahler Order'
            ylabel = 'Mean Diameter (mm)'
            plt.ylim((0, 3))
            plt.yticks(np.array(range(0, 30, 5))/10)
        elif content == "Length":
            mean_l = self.mean_branch_length()
            raw_data = {'x': range(1, self.max_HS + 1), 'y': np.array(mean_l)}
            df = pd.DataFrame(raw_data, index=range(1, self.max_HS + 1))
            title = 'Branch Length v.s. Strahler Order'
            ylabel = 'Mean Branch Length (mm)'
            plt.ylim((0, 40))
            plt.yticks(range(0, 40, 5))

        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("ticks")
        sns.lmplot('x', 'y', data=df, fit_reg=True, scatter_kws={"marker": "D", "s": 100})
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        
        if save:
            path = '/Users/kimihirochin/Desktop/mesh/plots/test_1_scatter_%d_%s.png' % (self.file_id, title)
            plt.savefig(path)
        else:
            plt.show()

    def boxplot(self, save=False):
        raw_data = []
        for i in range(len(self.connections)):
            hs = self.HS[i] * 2
            r = self.radii[i]
            raw_data.append([hs, r])
        df = pd.DataFrame(raw_data, columns=['Strahler Order', 'Radius'])
        title = 'Diameter v.s. Strahler Order'
        ylabel = 'Mean Diameter (mm)'

        sns.boxplot(x=df['Strahler Order'], y=df['Radius'])
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        
        if save:
            path = '/Users/kimihirochin/Desktop/mesh/plots/test_1_box_%d_%s.png' % (self.file_id, title)
            plt.savefig(path)
        else:
            plt.show()

    def plot(self):
        save = False
        self.histogram("Length", save)
        self.histogram("Log Length", save)
        self.histogram("Diameter", save)
        self.histogram("Log Diameter", save)
        self.scatterplot("Length", save)
        self.scatterplot("Diameter", save)
        self.boxplot()


if __name__ == '__main__':
    file_id = 12
    coord_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_coords.npy' % file_id
    connection_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_connections.npy' % file_id
    radius_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_radii.npy' % file_id
    HS_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_HS_order.npy' % file_id
    level_file = '/Users/kimihirochin/Desktop/mesh/test_1_result_%d_level_order.npy' % file_id
    analyzer = GCOAnalyzer(coord_file, connection_file, radius_file, HS_file, level_file, file_id)
    # analyzer.plot()
    analyzer.histogram("Diameter")
    analyzer.histogram("Log Diameter")
    analyzer.histogram("Length")
    analyzer.histogram("Log Length")
    
    analyzer.scatterplot("Length")
    analyzer.scatterplot("Diameter")
    # analyzer.analyze_strahler()
    # analyzer.reconstruct_model()