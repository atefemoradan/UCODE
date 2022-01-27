from UCODEncoder import GCN
import utils as com
from sklearn.cluster import KMeans
import pickle as pkl
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from Lossfunction import loss_modularity_trace
from scipy import sparse
import os
from sklearn import metrics
from utils import precision
from utils import recall
from matplotlib import pyplot as plt

from networkx.generators.degree_seq import expected_degree_graph

N_COMMUNITY_DICT = {
    'cora': 7,
    'fb_1684': 17,
    'fb_1912': 46
}

class OverlapAnalyzer:
    def __init__(
        self,
        epochs=500,
        hid_units=16,
        overlap_name='fb_1684',
        disjoint_name='cora'
    ):
        self.path = os.path.dirname(os.path.abspath(__file__))

        self.disjoint_name = disjoint_name
        self.disjoint_data = {}
        self.overlap_name = overlap_name
        self.overlap_data = {}

        # FIXME FIXME FIXME -- what is the n_communities of each dataset??
        # It's the length of the one-hot, yes?
        self.hid_units = hid_units
        self.epochs = epochs
        self.x_ent = nn.BCEWithLogitsLoss()

        self._get_disjoint_data()
        self._get_overlap_data()
        self.cuda()

    def cuda(self):
        if torch.cuda.is_available():
            for data in [self.disjoint_data, self.overlap_data]:
                data['features'] = data['features'].cuda()
                data['B'] = data['B'].cuda()
                data['adj'] = data['adj'].cuda()

    def _get_disjoint_data(self):
        adj, features, labels = com.load_data('cora')
        graph_file = 'ind.' + self.disjoint_name + '.graph'
        with open(os.path.join(self.path, 'dataset', graph_file), 'rb') as f:
            graph = pkl.load(f, encoding='latin1')
        nb_nodes, ft_size = features.shape

        mod_filename = 'Modularity-' + self.disjoint_name + '.npy'
        B = np.load(os.path.join(self.path, 'dataset', mod_filename))
        features, adj, B, labels_clus = com.convert_torch_kipf(adj, features, B, labels)

        try:
            self.disjoint_data['n_communities'] = N_COMMUNITY_DICT[self.disjoint_name]
        except KeyError:
            raise ValueError('Dataset %s is not in the N_COMMUNITY_DICT' % self.disjoint_name)
        self.disjoint_data['adj'] = adj
        self.disjoint_data['features'] = features
        self.disjoint_data['labels'] = labels_clus
        self.disjoint_data['B'] = B
        self.disjoint_data['m'] = len(nx.Graph(graph).edges)
        self.disjoint_data['nb_nodes'] = nb_nodes
        self.disjoint_data['ft_size'] = ft_size

    def _get_overlap_data(self):
        npz_path = os.path.join(self.path, 'dataset', self.overlap_name + '.npz')
        sparse_graph = com.load_npz_dataset(npz_path)
        adj = sparse_graph['A']
        features = sparse.csr_matrix(sparse_graph['X'])
        labels = sparse_graph['Z']
        m = len(nx.from_scipy_sparse_matrix(adj).edges)
        nb_nodes, ft_size = features.shape

        mod_filename = 'Modularity-' + self.overlap_name + '.npy'
        B = np.load(os.path.join(self.path, 'dataset', mod_filename))
        features, adj, B, _ = com.convert_torch_npz(adj, features, B, labels)

        try:
            self.overlap_data['n_communities'] = N_COMMUNITY_DICT[self.overlap_name]
        except KeyError:
            raise ValueError('Dataset %s is not in the N_COMMUNITY_DICT' % self.overlap_name)
        self.overlap_data['adj'] = adj
        self.overlap_data['features'] = features
        self.overlap_data['labels'] = labels
        self.overlap_data['B'] = B
        self.overlap_data['m'] = m
        self.overlap_data['nb_nodes'] = nb_nodes
        self.overlap_data['ft_size'] = ft_size

    def base_loss(self, logits, data):
        logits_T = logits.transpose(1, 2)
        modularity = torch.matmul(torch.matmul(logits_T, data['B']), logits)

        idx = np.random.permutation(self.hid_units)
        shuf_fts = modularity[:, idx, :]
        lbl_1 = torch.ones(1, self.hid_units)
        lbl_2 = torch.zeros(1, self.hid_units)
        lbl = torch.cat((lbl_2, lbl_1), 1)
        diag1 = torch.div(torch.diag(modularity[0], 0), data['m'])
        diag2 = torch.div(torch.diag(shuf_fts[0], 0), data['m'])

        diags = torch.cat((diag2, diag1), 0)
        diags = diags.unsqueeze(0)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
            diags = diags.cuda()

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
        loss = self.x_ent(diags, lbl)
        loss.requires_grad_(True)
        return loss

    def sort_loss(self, logits, data, contrast_value):
        logits_T = logits.transpose(1, 2)
        modularity = torch.matmul(torch.matmul(logits_T, data['B']), logits)

        inds = torch.argsort(modularity, dim=1)
        zeros = torch.zeros_like(modularity)
        mod2 = modularity * (inds == contrast_value)
        print(torch.max(mod2, dim=1))
        quit()
        shuf_fts = modularity[:, idx, :]
        lbl_1 = torch.ones(1, self.hid_units)
        lbl_2 = torch.zeros(1, self.hid_units)
        lbl = torch.cat((lbl_2, lbl_1), 1)
        diag1 = torch.div(torch.diag(modularity[0], 0), data['m'])
        diag2 = torch.div(torch.diag(shuf_fts[0], 0), data['m'])

        diags = torch.cat((diag2, diag1), 0)
        diags = diags.unsqueeze(0)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
            diags = diags.cuda()

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
        loss = self.x_ent(diags, lbl)
        loss.requires_grad_(True)
        return loss

    def analyze_overlap(self):
        scores = {
            'nmit': [],
            'modularityt': [],
            'conductancet': [],
            'cst': [],
            'precisiont': [],
            'recalt': [],
            'Fscoret': [],
            'nmikmean': [],
        }
        for i in range(1):
            for data in [self.disjoint_data, self.overlap_data]:
                model = None
                model = GCN(data['ft_size'], self.hid_units, data['nb_nodes'])
                if torch.cuda.is_available():
                    model.cuda()
                optimiser = torch.optim.Adam(
                    model.parameters(),
                    lr=0.001,
                    weight_decay=0.001
                )
                model.train()
                for epoch in range(self.epochs):
                    optimiser.zero_grad()

                    logits = model(data['features'], data['adj'])
                    loss = self.sort_loss(logits, data, 1)
                    loss.backward()
                    optimiser.step()

                    embeds = logits.view(
                        data['nb_nodes'],
                        self.hid_units
                    ).detach().cpu().numpy()
                    kmeans = KMeans(
                        init='k-means++',
                        n_clusters=data['n_communities'],
                        random_state=0
                    ).fit(embeds)
                    nmi_kmean = normalized_mutual_info_score(data['labels'], kmeans.labels_)
                    print(epoch, '---', nmi_kmean)

                modularity = com.modularity(data['adj'], kmeans.labels_)
                conductance = com.conductance(data['adj'], kmeans.labels_)
                p = precision(data['labels'], kmeans.labels_)
                r = recall(data['labels'], kmeans.labels_)
                completeness = metrics.completeness_score(data['labels'], kmeans.labels_)
                F_score = 2 * ((r * p) / (r + p))


if __name__ == '__main__':
    analyzer = OverlapAnalyzer()
    analyzer.analyze_overlap()
