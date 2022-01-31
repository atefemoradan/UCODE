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
import copy
from sklearn import metrics
from utils import precision
from utils import recall
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

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
        num_iters=16,
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
        self.num_iters = num_iters
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

        lbl = torch.cat((
            torch.ones(1, self.hid_units),
            torch.zeros(1, self.hid_units)
        ), 1)

        attractive = torch.div(torch.diag(modularity[0], 0), data['m'])
        inds = torch.argsort(modularity, dim=1)
        sort_mask = inds == contrast_value
        repulsive = torch.masked_select(modularity, sort_mask)
        repulsive = torch.div(repulsive, data['m'])

        diags = torch.cat((attractive, repulsive), 0)
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

    def _get_empty_dicts(self):
        dataset_scores = {
            'nmi': [],
            'modularity': [],
            'conductance': [],
            'completeness': [],
            'precision': [],
            'recall': [],
            'Fscore': [],
        }
        scores_dict = {
            'disjoint': copy.deepcopy(dataset_scores),
            'overlap': copy.deepcopy(dataset_scores)
        }

        dataset_dict = {
            'disjoint': self.disjoint_data,
            'overlap': self.overlap_data
        }
        return scores_dict, dataset_dict

    def analyze_overlap(self):
        scores_dict, dataset_dict = self._get_empty_dicts()
        for i in range(1, self.num_iters):
            for mode in ['disjoint', 'overlap']:
                data = dataset_dict[mode]
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
                    loss = self.sort_loss(logits, data, i)
                    # loss = self.base_loss(logits, data)
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

                np_adj = torch.squeeze(data['adj']).detach().numpy()
                modularity = com.modularity(np_adj, kmeans.labels_)
                conductance = com.conductance(np_adj, kmeans.labels_)
                p = precision(data['labels'], kmeans.labels_)
                r = recall(data['labels'], kmeans.labels_)
                completeness = metrics.completeness_score(data['labels'], kmeans.labels_)
                F_score = 2 * ((r * p) / (r + p))

                scores_dict[mode]['nmi'].append(nmi_kmean)
                scores_dict[mode]['modularity'].append(modularity)
                scores_dict[mode]['conductance'].append(conductance)
                scores_dict[mode]['completeness'].append(completeness)
                scores_dict[mode]['precision'].append(p)
                scores_dict[mode]['recall'].append(r)
                scores_dict[mode]['Fscore'].append(F_score)

        np.save('overlap_analysis_results.npy', scores_dict)

def make_plots(path, num_iters):
    scores_dict = np.load(path, allow_pickle=True)[()]
    x_vals = np.arange(1, num_iters)
    modes = ['disjoint', 'overlap']
    cmap = dict(zip(modes, ['b', 'g']))
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]
    for i, key in enumerate(scores_dict['disjoint']):
        disjoint_scores = scores_dict['disjoint'][key]
        overlap_scores = scores_dict['overlap'][key]

        ax = plt.subplot(111)
        ax.bar(x_vals - 0.15, disjoint_scores, width=0.3, color='b', align='center')
        ax.bar(x_vals + 0.15, overlap_scores, width=0.3, color='g', align='center')
        ax.set_ylim([0.0, max([max(disjoint_scores), max(overlap_scores)]) * 1.2])
        plt.title('Effect of contrastive index on %s' % key)
        plt.legend(
            labels=modes,
            handles=patches,
            loc='upper center',
            ncol=num_iters - 1,
            borderaxespad=0,
            fontsize=15,
        )
        plt.xticks(x_vals)
        plt.xlabel('Contrastive index')
        plt.ylabel(key)

        plt.savefig(os.path.join('analysis_figs', '{}.png'.format(key)))
        plt.clf()

if __name__ == '__main__':
    # analyzer = OverlapAnalyzer(num_iters=16)
    # analyzer.analyze_overlap()

    make_plots('overlap_analysis_results.npy', 16)
