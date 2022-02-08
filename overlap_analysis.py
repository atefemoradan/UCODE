from UCODEncoder import GCN
import utils as com
from sklearn.cluster import KMeans
import pickle as pkl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from Lossfunction import loss_modularity_trace
from scipy import sparse
import os
import copy
from sklearn import metrics
from utils import precision
from utils import disjoint_recall, overlap_recall, overlap_nmi
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
        hid_units=16,
        overlap_name='fb_1684',
        disjoint_name='cora'
    ):
        self.path = os.path.dirname(os.path.abspath(__file__))

        self.disjoint_name = disjoint_name
        self.disjoint_data = {}
        self.overlap_name = overlap_name
        self.overlap_data = {}

        self.epochs = epochs
        self.x_ent = nn.BCELoss()

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
        self.disjoint_data['hid_units'] = labels.shape[-1]
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
        self.overlap_data['hid_units'] = labels.shape[-1]
        self.overlap_data['ft_size'] = ft_size

    def loss(self, logits, data):
        logits_T = logits.transpose(1, 2)
        modularity = torch.matmul(torch.matmul(logits_T, data['B']), logits)

        # ANDREW - Uncomment this to make modularity matrix > 0 for each element
        # modularity -= torch.min(modularity, dim=1, keepdims=True)[0]

        idx = np.random.permutation(data['hid_units'])
        shuf_fts = modularity[:, idx, :]
        lbl_1 = torch.ones(1, data['hid_units'])
        lbl_2 = torch.zeros(1, data['hid_units'])
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

        # ANDREW - this is the loss defined on line 45
        loss = self.x_ent(diags, lbl)

        loss.requires_grad_(True)
        return loss

    def _get_empty_dicts(self):
        dataset_scores = {'nmi': [], 'recall': []}
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
        for mode in ['disjoint', 'overlap']:
            data = dataset_dict[mode]
            for i in (list(range(16, data['hid_units'], 8)) + [62]):
                model = GCN(data['ft_size'], data['hid_units'], data['nb_nodes'])
                if torch.cuda.is_available():
                    model.cuda()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.001,
                    weight_decay=0.001
                )
                lr_sched = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=75,
                    gamma=0.5,
                )
                model.train()
                for epoch in range(self.epochs + 1):
                    optimizer.zero_grad()

                    logits = model(data['features'], data['adj'])
                    loss = self.loss(torch.exp(logits), data)
                    loss.backward()
                    optimizer.step()

                    embeds = logits.view(
                        data['nb_nodes'],
                        data['hid_units']
                    ).detach().cpu().numpy()
                    if mode == 'disjoint':
                        # kmeans = KMeans(
                        #     init='k-means++',
                        #     n_clusters=data['n_communities'],
                        #     random_state=0
                        # ).fit(embeds)
                        # preds = kmeans.labels_
                        preds = np.argmax(embeds, axis=-1)
                        nmi = normalized_mutual_info_score(data['labels'], preds)
                        recall = disjoint_recall(data['labels'], preds)
                    else:
                        logits = logits.cpu().detach().numpy()
                        # thresh = np.mean(logits, axis=-1, keepdims=True)
                        thresh = np.mean(logits, axis=-1, keepdims=True)
                        preds =  np.squeeze(logits > thresh)
                        nmi = overlap_nmi(data['labels'], preds)
                        recall = overlap_recall(
                            data['labels'],
                            preds,
                            data['n_communities']
                        )
                    lr_sched.step()
                    if epoch % 25 == 0:
                        np_loss = loss.cpu().detach().numpy()
                        output = '--- {} loss: {:.03f} - nmi: {:.04f} - recall: {:.04f}'
                        print(epoch, output.format(mode, np_loss, nmi, recall))

                scores_dict[mode]['nmi'].append(nmi)
                scores_dict[mode]['recall'].append(recall)
                print('\n')

        np.save('overlap_analysis_results.npy', scores_dict)

def make_plots(path):
    scores_dict = np.load(path, allow_pickle=True)[()]
    modes = ['disjoint', 'overlap']
    for mode in modes:
        for i, key in enumerate(scores_dict[mode]):
            scores = scores_dict[mode][key]

            x_vals = np.arange(len(scores)) + 1
            ax = plt.subplot(111)
            ax.bar(
                x_vals,
                scores,
                width=0.3,
                color='b',
                align='center'
            )
            ax.set_ylim([min(scores) * 0.8, max(scores) * 1.2])

            plt.title('Effect of contrastive index on %s' % key)
            plt.xticks(x_vals)
            plt.xlabel('Contrastive index')
            plt.ylabel(key)

            plt.savefig(os.path.join('analysis_figs', '{}_{}.png'.format(mode, key)))
            plt.clf()

if __name__ == '__main__':
    analyzer = OverlapAnalyzer()
    analyzer.analyze_overlap()

    # make_plots('overlap_analysis_results.npy')
