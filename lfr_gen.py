import numpy as np
import os
import math
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=np.inf)

def draw_full_graph(
        g,
        node_str,
        edge_str,
        queried_str,
        node_cmap=plt.cm.gist_stern,
        edge_cmap=plt.cm.gist_stern,
        file_name='network.png',
        directory='figures'
    ):
    pos = nx.spring_layout(g, seed=123)
    node_attributes = np.array(list(nx.get_node_attributes(g, node_str).values()))
    queries = np.array(list(nx.get_node_attributes(g, queried_str).values()))
    edge_attributes = np.array(list(nx.get_edge_attributes(g, edge_str).values()))

    min_ea = min(edge_attributes)
    max_ea = max(edge_attributes)
    fig = plt.figure(figsize=(12, 10))
    ec = nx.draw_networkx_edges(
        g,
        pos,
        alpha=0.5,
        width=[2 * ea / max_ea for ea in edge_attributes],
        edge_color=edge_attributes,
        edge_cmap=edge_cmap,
        edge_vmin=min_ea,
        edge_vmax=max_ea
    )
    ecb = plt.colorbar(ec)
    ecb.set_label(edge_str)

    max_na = max(node_attributes)
    queried_nodelist = np.array(g.nodes())[np.where(queries == 1)]
    queried_attributes = node_attributes[np.where(queries == 1)]
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=queried_nodelist,
        node_color=queried_attributes, 
        node_size=100,
        edgecolors=[1, 0, 0],
        linewidths=2,
        cmap=node_cmap,
        vmin=0,
        vmax=max_na
    )
    other_nodelist = np.array(g.nodes())[np.where(queries == 0)]
    other_attributes = node_attributes[np.where(queries == 0)]
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=other_nodelist,
        node_color=other_attributes, 
        node_size=100,
        cmap=node_cmap,
        vmin=0,
        vmax=max_na
    )

    ncb = plt.colorbar(nc)
    ncb.set_label(node_str)
    plt.axis('off')
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()
    plt.close()



class LFRGenerator:
    """

    """
    def __init__(
        self,
        n,
        min_degree,
        average_degree,
        max_degree,
        min_community,
        max_community,
        tau1,
        tau2,
        mu,
        seed=20,
        presaved_path=None
    ):
        self.n = n

        self.min_degree = min_degree
        self.average_degree = average_degree
        self.max_degree = max_degree
        self.min_community = min_community
        self.max_community = max_community
        self.tau1 = tau1
        self.tau2 = tau2
        self.mu = mu

        self.labels = {}
        self.community_sizes = {}

        self.name = '_'.join("%s:%s" % item for item in vars(self).items())
        self.plot_name = self.name + '.png'
        self.plot_dir = 'LFR_figures'
        self.file_name = self.name + '.dat'
        self.file_dir = 'LFR_graphs'

        if presaved_path is not None:
            print('Loaded graph from %s' % presaved_path)
            self.graph = nx.readwrite.adjlist.read_adjlist(presaved_path)
        else:
            print('Generating LFR graph...')
            self.graph = LFR_benchmark_graph(
                self.n,
                self.tau1,
                self.tau2,
                self.mu,
                average_degree=self.average_degree,
                min_community=self.min_community,
                max_community=self.max_community,
                seed=seed
            )

    def process_communities(self):
        communities = list(nx.get_node_attributes(self.graph, 'community').values())
        communities = [list(member) for member in communities]
        processed_comms = []
        label_counter = 0
        for _, community in enumerate(communities):
            if hash(tuple(community)) not in processed_comms:
                processed_comms += [hash(tuple(community))]
                comm_size = len(community)
                for node in community:
                    assert node not in self.labels
                    self.labels[node] = label_counter
                    self.community_sizes[node] = comm_size
                label_counter += 1
        nx.set_node_attributes(self.graph, self.labels, 'labels')
        nx.set_node_attributes(self.graph, self.community_sizes, 'community_sizes')


    def save_graph(self, path=None):
        if not os.path.isdir(self.file_dir):
            os.makedirs(self.file_dir)
        if path is None:
            path = os.path.join(self.file_dir, self.file_name)
        nx.readwrite.adjlist.write_adjlist(self.graph, path)

    def save_plot(self, path=None):
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
        if path is None:
            path = os.path.join(self.plot_dir, self.plot_name)
        self._save_plot(path)

    def draw_plot(self):
        # drawing nodes and edges separately so we can capture collection for colobar
        print('\tCalculating spring layout for plot...')
        fig = plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph, seed=123, k=0.35)
        ec = nx.draw_networkx_edges(self.graph, pos, alpha=0.1)
        nc = nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self.graph.nodes(),
            node_color=list(self.labels.values()), 
            node_size=20,
            cmap=plt.cm.PiYG,
            vmin=0,
            vmax=max(list(self.labels.values()))
        )
        cb = plt.colorbar(nc)
        cb.set_label('Community Label')
        plt.axis('off')
        plt.show()
        # plt.savefig('figures/community_membership.png')
        plt.clf()

    def _save_plot(self, path):
        pass


if __name__ == '__main__':
    lfr = LFRGenerator(
        n=1000,
        min_degree=3,
        average_degree=7,
        max_degree=30,
        min_community=3,
        max_community=30,
        tau1=3,
        tau2=1.5,
        mu=0.4,
    )
    lfr.process_communities()
    lfr.draw_plot()
