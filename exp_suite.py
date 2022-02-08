# out-of-library imports
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
import pickle as pkl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy import sparse
import os
from sklearn import metrics
from networkx.generators.degree_seq import expected_degree_graph

# In-library imports
from UCODEncoder import GCN
from utils import disjoint_recall, overlap_recall, overlap_nmi
import utils as com

N_COMMUNITY_DICT = {
    'cora': 7,
    'citeseer': 6,
    'pubmed': 3,
    'amazon_photo': 8,
    'amazon_pc': 10,
    'coauthor_cs': 15,
    'fb_348': 14,
    'fb_414': 7,
    'fb_686': 14,
    'fb_698': 13,
    'fb_1684': 17,
    'fb_1912': 46
}

class ExperimentRunner:
    def __init__(
        self,
        disjoint_datasets,
        overlap_datasets,
        epochs=500,
        hid_units=16,
    ):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.pd_dir = 'dataframes'
        os.makedirs(self.pd_dir, exist_ok=True)

        self.disjoint_datasets = disjoint_datasets
        self.disjoint_data = {}
        self.overlap_datasets = overlap_datasets
        self.overlap_data = {}

        self.epochs = epochs
        self.x_ent = nn.BCEWithLogitsLoss()

        self._get_disjoint_data()
        self._get_overlap_data()
        self.datasets = {
            'disjoint': self.disjoint_data,
            'overlap': self.overlap_data
        }
        self.cuda()

    def cuda(self):
        if torch.cuda.is_available():
            for data in [self.disjoint_data, self.overlap_data]:
                data['features'] = data['features'].cuda()
                data['B'] = data['B'].cuda()
                data['adj'] = data['adj'].cuda()

    def _get_B(self, mod_path, nx_graph):
        if not os.path.exists(mod_path):
            print('Creating B matrix for %s' % mod_path)
            B = com.get_B(nx_graph)
            np.save(mod_path, B)
        else:
            B = np.load(mod_path)

        return B

    def _get_disjoint_data(self):
        for disjoint_name in self.disjoint_datasets:
            self.disjoint_data[disjoint_name] = {}

            adj, features, labels = com.load_data(disjoint_name)
            graph_file = 'ind.' + disjoint_name + '.graph'
            with open(os.path.join(self.path, 'dataset', graph_file), 'rb') as f:
                graph = pkl.load(f, encoding='latin1')
            nb_nodes, ft_size = features.shape
            nx_graph = nx.from_scipy_sparse_matrix(adj)

            mod_filename = 'Modularity-' + disjoint_name + '.npy'
            mod_path = os.path.join(self.path, 'dataset', mod_filename)
            B = self._get_B(mod_path, nx_graph)
            features, adj, B, labels_clus = com.convert_torch_kipf(adj, features, B, labels)

            try:
                self.disjoint_data[disjoint_name]['n_communities'] = N_COMMUNITY_DICT[disjoint_name]
            except KeyError:
                raise ValueError('Dataset %s is not in the N_COMMUNITY_DICT' % disjoint_name)
            self.disjoint_data[disjoint_name]['adj'] = adj
            self.disjoint_data[disjoint_name]['features'] = features
            self.disjoint_data[disjoint_name]['labels'] = labels_clus
            self.disjoint_data[disjoint_name]['B'] = B
            self.disjoint_data[disjoint_name]['m'] = len(nx.Graph(graph).edges)
            self.disjoint_data[disjoint_name]['nb_nodes'] = nb_nodes
            self.disjoint_data[disjoint_name]['hid_units'] = 64 #labels.shape[-1]
            self.disjoint_data[disjoint_name]['ft_size'] = ft_size
            self.disjoint_data[disjoint_name]['nx_graph'] = nx_graph

    def _get_overlap_data(self):
        for overlap_name in self.overlap_datasets:
            self.overlap_data[overlap_name] = {}

            npz_path = os.path.join(self.path, 'dataset', overlap_name + '.npz')
            sparse_graph = com.load_npz_dataset(npz_path)
            adj = sparse_graph['A']
            features = sparse.csr_matrix(sparse_graph['X'])
            labels = sparse_graph['Z']
            nx_graph = nx.from_scipy_sparse_matrix(adj)
            m = len(nx_graph.edges)
            nb_nodes, ft_size = features.shape

            mod_filename = 'Modularity-' + overlap_name + '.npy'
            mod_path = os.path.join(self.path, 'dataset', mod_filename)
            B = self._get_B(mod_path, nx_graph)

            features, adj, B, _ = com.convert_torch_npz(adj, features, B, labels)

            try:
                self.overlap_data[overlap_name]['n_communities'] = N_COMMUNITY_DICT[overlap_name]
            except KeyError:
                raise ValueError('Dataset %s is not in the N_COMMUNITY_DICT' % overlap_name)
            self.overlap_data[overlap_name]['adj'] = adj
            self.overlap_data[overlap_name]['features'] = features
            self.overlap_data[overlap_name]['labels'] = labels
            self.overlap_data[overlap_name]['B'] = B
            self.overlap_data[overlap_name]['m'] = m
            self.overlap_data[overlap_name]['nb_nodes'] = nb_nodes
            self.overlap_data[overlap_name]['hid_units'] = labels.shape[-1]
            self.overlap_data[overlap_name]['ft_size'] = ft_size
            self.overlap_data[overlap_name]['nx_graph'] = nx_graph

    def loss(self, logits, data):
        logits_T = logits.transpose(1, 2)
        modularity = torch.matmul(torch.matmul(logits_T, data['B']), logits)

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
        loss = self.x_ent(diags, lbl)
        loss.requires_grad_(True)
        return loss

    def _get_dataset_metrics(self):
        disjoint_metrics = {i: {} for i in self.disjoint_data}
        overlap_metrics = {i: {} for i in self.overlap_data}
        metrics = {}
        metric_names = [
            'dataset_name',
            'overlapping',
            'edge_density',
            'min_degree',
            'max_degree',
            'degree_mean',
            'degree_variance',
            'mean_community_size',
            'biggest_community_index',
            'size_of_biggest_community',
            'smallest_community_index',
            'size_of_smallest_community',
        ]
        for m in metric_names:
            metrics[m] = {}
        for dataset_group in self.datasets:
            for dataset in self.datasets[dataset_group]:
                nx_graph = self.datasets[dataset_group][dataset]['nx_graph']
                # Edge statistics
                degrees = np.array([i[1] for i in nx_graph.degree])
                metrics['overlapping'][dataset] = dataset_group == 'overlap'
                metrics['dataset_name'][dataset] = dataset
                metrics['edge_density'][dataset] = nx.density(nx_graph)
                metrics['min_degree'][dataset] = np.min(degrees)
                metrics['max_degree'] [dataset] = np.max(degrees)
                metrics['degree_mean'][dataset] = np.mean(degrees)
                metrics['degree_variance'][dataset] = np.var(degrees)

                # Label statistics
                labels = self.datasets[dataset_group][dataset]['labels']
                if dataset_group == 'disjoint':
                    # Labels are easy to bin due to not overlapping
                    comm_sizes = np.bincount(labels)
                    metrics['mean_community_size'][dataset] = np.mean(comm_sizes)
                    metrics['biggest_community_index'][dataset] = np.argmax(comm_sizes)
                    metrics['size_of_biggest_community'][dataset] = int(np.max(comm_sizes))
                    metrics['smallest_community_index'][dataset] = np.argmin(comm_sizes)
                    metrics['size_of_smallest_community'][dataset] = int(np.min(comm_sizes))
                else:
                    # In overlapping case, have to first disentangle each label
                    comm_sizes = np.sum(labels, axis=0)
                    metrics['mean_community_size'][dataset] = np.mean(comm_sizes)
                    metrics['biggest_community_index'][dataset] = np.argmax(comm_sizes)
                    metrics['size_of_biggest_community'][dataset] = int(np.max(comm_sizes))
                    metrics['smallest_community_index'][dataset] = np.argmin(comm_sizes)
                    metrics['size_of_smallest_community'][dataset] = int(np.min(comm_sizes))

        return metrics

    def plot_dataset_metrics(self):
        dataset_metrics = self._get_dataset_metrics()
        dataset_metrics = pd.DataFrame(dataset_metrics)
        dataset_metrics.to_csv(os.path.join(self.pd_dir, 'dataset_metrics.csv'))
        fig = px.scatter(
            dataset_metrics,
            x="degree_mean",
            y="degree_variance",
            color="overlapping",
            size='mean_community_size',
            hover_data=['dataset_name', 'edge_density', 'min_degree', 'max_degree']
        )
        fig.write_html(os.path.join('plotly', 'degree_stats.html'))

        fig = px.scatter(
            dataset_metrics,
            x="size_of_smallest_community",
            y="size_of_biggest_community",
            color="overlapping",
            size='mean_community_size',
            hover_data=['dataset_name', 'biggest_community_index', 'smallest_community_index']
        )
        fig.write_html(os.path.join('plotly', 'label_stats.html'))

    def run_GCN(self):
        results = {
            'modularity': {},
            'nmi': {},
            'recall': {},
            'precision': {},
            'Fscore': {},
            'conductance': {},
            'completeness': {}
        }
        for dataset_group in self.datasets: # first disjoint then overlapping
            for dataset in self.datasets[dataset_group]:
                data = self.datasets[dataset_group][dataset]
                # Set up the model
                model = GCN(data['ft_size'], data['hid_units'], data['nb_nodes'])
                if torch.cuda.is_available():
                    model.cuda()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.002,
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
                    loss = self.loss(logits, data)
                    loss.backward()
                    optimizer.step()

                    embeds = logits.view(
                        data['nb_nodes'],
                        data['hid_units']
                    ).detach().cpu().numpy()
                    if dataset_group == 'disjoint':
                        kmeans = KMeans(
                            init='k-means++',
                            n_clusters=data['n_communities'],
                            random_state=0
                        ).fit(embeds)
                        preds = kmeans.labels_
                        nmi = normalized_mutual_info_score(data['labels'], preds)
                    else:
                        logits = logits.cpu().detach().numpy()
                        # XXX - this is different from the threshold in Atefeh's model
                        #     - I found that it achieves better scores on overlapping NMI
                        thresh = np.mean(logits, axis=-1, keepdims=True)
                        preds =  np.squeeze(logits > thresh)
                        nmi = overlap_nmi(data['labels'], preds)
                    lr_sched.step()
                    if epoch % 25 == 0:
                        np_loss = loss.cpu().detach().numpy()
                        output = '--- {} loss: {:.03f} - nmi: {:.04f}'
                        print(epoch, output.format(dataset_group, np_loss, nmi))

                model.eval()
                # Get model results at the end of training
                if dataset_group == 'disjoint':
                    # These metrics only make sense for non-overlapping
                    recall = disjoint_recall(data['labels'], preds)
                    np_adj = torch.squeeze(data['adj']).detach().numpy()
                    modularity = com.modularity(np_adj, kmeans.labels_)
                    conductance = com.conductance(np_adj, kmeans.labels_)
                    precision = com.precision(data['labels'], kmeans.labels_)
                    completeness = metrics.completeness_score(data['labels'], kmeans.labels_)
                    F_score = 2 * ((recall * precision) / (recall + precision))

                    results['modularity'][dataset] = modularity
                    results['conductance'][dataset] = conductance
                    results['completeness'][dataset] = completeness
                    results['precision'][dataset] = precision
                    results['Fscore'][dataset] = F_score
                else:
                    recall = overlap_recall(
                        data['labels'],
                        preds,
                        data['n_communities']
                    )


                results['nmi'][dataset] = nmi
                results['recall'][dataset] = recall

                print('Results for GCN on %s dataset:' % dataset)
                print(pd.DataFrame(results).loc[[dataset]])
                print('\n')

        results = pd.DataFrame(results)
        results.to_csv(os.path.join(self.pd_dir, 'GCN_results.csv'))


def plot_relationship(dataset_metric, results_metric, hover_variables=[]):
    results_df_path = os.path.join('dataframes', 'GCN_results.csv')
    dataset_df_path = os.path.join('dataframes', 'dataset_metrics.csv')
    results_df = pd.read_csv(results_df_path)
    dataset_df = pd.read_csv(dataset_df_path)

    # Get requested metrics
    try:
        overlapping = pd.DataFrame(dataset_df['overlapping'])
        dataset = pd.DataFrame(dataset_df['dataset_name'])
    except KeyError:
        raise ValueError("No information on overlap in {}".format(dataset_df_path))
    try:
        results = pd.DataFrame(results_df[results_metric])
    except KeyError:
        raise ValueError("Metric {} is not available in {}".format(results_metric, results_df_path))
    try:
        data_metric = pd.DataFrame(dataset_df[dataset_metric])
    except KeyError:
        raise ValueError("Metric {} is not available in {}".format(dataset_metric, dataset_df_path))

    # Get auxiliary metrics
    hover_data = None
    for hover_var in hover_variables:
        try:
            var = pd.DataFrame(results_df[hover_var])
        except KeyError:
            var = pd.DataFrame(dataset_df[hover_var])
        except:
            raise ValueError("Metric {} is not available in either {} or {}".format(hover_var, results_df_path, dataset_df_path))

        if hover_data is None:
            hover_data = var
        else:
            hover_data = hover_data.join(var)
    df = overlapping.join(results).join(data_metric).join(hover_data).join(dataset)

    fig = px.scatter(
        df,
        x=dataset_metric,
        y=results_metric,
        color="overlapping",
        hover_data=hover_variables + ['dataset_name']
    )
    fig.write_html(os.path.join('plotly', '{}_vs_{}.html'.format(results_metric, dataset_metric)))

if __name__ == '__main__':
    analyzer = ExperimentRunner(
        disjoint_datasets=['cora', 'citeseer'],
        overlap_datasets=['fb_1684', 'fb_1912']
    )
    analyzer.plot_dataset_metrics()

    # Comment this out to train the GCN on each dataset
    # This will save off the resulting metrics into dataframes/GCN_results.csv
    # analyzer.run_GCN()

    plot_relationship('degree_mean', 'nmi', hover_variables=['min_degree', 'edge_density'])
