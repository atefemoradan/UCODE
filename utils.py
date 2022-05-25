
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from sklearn.cluster import KMeans
import random
from sklearn.cluster import SpectralClustering
from itertools import product
import scipy.sparse as sp
import sys
import torch
import os
from sklearn.metrics.cluster import contingency_matrix
from scipy.sparse import csr_matrix

path=os.path.dirname(os.path.abspath(__file__))+'/venv/'
# From Dmon Model
def _compute_counts(y_true, y_pred):    #
    contingency = contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (
            total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives


def precision(y_true, y_pred):
    true_positives, false_positives, _, _ = _compute_counts(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives, _, false_negatives, _ = _compute_counts(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
                index.append(int(line.strip()))
        return index

def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
                with open(path+"/dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                        if sys.version_info > (3, 0):
                                objects.append(pkl.load(f, encoding='latin1'))
                        else:
                                objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(path+"/dataset/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
                # Fix citeseer dataset (there are some isolated nodes in the graph)
                # Find isolated nodes, add them as zero-vecs into the right position
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended
                ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                ty_extended[test_idx_range-min(test_idx_range), :] = ty
                ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

### Think about other variables

        return adj, features, labels


def CKmeans(features,nb_community):

        kmeans = KMeans(init='k-means++', n_clusters=nb_community, random_state=0).fit(features)
        labels= kmeans.labels_
        return labels

def random_edge(graph):

        edges = list(graph.edges)
        nonedges = list(nx.non_edges(graph))

        # random edge choice
        chosen_edge = random.choice(edges)
        chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0]])

        # add new edge
        graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

        return graph
"""Calculate B Matrix"""

def get_B(G):
    Q = 0
    G = G.copy()
    nx.set_edge_attributes(G, {e:1 for e in G.edges}, 'weight')
    A = nx.to_scipy_sparse_matrix(G).astype(float)

    if type(G) == nx.Graph:
            # for undirected graphs, in and out treated as the same thing
            out_degree = in_degree = dict(nx.degree(G))
            print(out_degree)
            M = 2.*(G.number_of_edges())
            print("Calculating modularity for undirected graph")
    elif type(G) == nx.DiGraph:
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            M = 1.*G.number_of_edges()
            print("Calculating modularity for directed graph")
    else:
            print('Invalid graph type')
            raise TypeError
    Q=np.zeros(A.shape)
    nodes = list(G)
    for i, j in product(range(len(nodes)),range(len(nodes))):
        Q[i,j]=A[i,j]-(in_degree[nodes[i]]*out_degree[nodes[j]]/M )
    return Q/4


def sparse_to_tuple(sparse_mx, insert_batch=False):
        """Convert sparse matrix to tuple representation."""
        """Set insert_batch=True if you want to insert a batch dimension."""
        def to_tuple(mx):
                if not sp.isspmatrix_coo(mx):
                        mx = mx.tocoo()
                if insert_batch:
                        coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
                        values = mx.data
                        shape = (1,) + mx.shape
                else:
                        coords = np.vstack((mx.row, mx.col)).transpose()
                        values = mx.data
                        shape = mx.shape
                return coords, values, shape

def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

""""The followings are fromDMON paper"""
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def conductance(adjacency, clusters):
    inter = 0
    intra = 0
    cluster_idx = np.zeros(adjacency.shape[0], dtype=np.bool)
    for cluster_id in np.unique(clusters):
        cluster_idx[:] = 0
        cluster_idx[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_idx, :]
        inter += np.sum(adj_submatrix[:, cluster_idx])
        intra += np.sum(adj_submatrix[:, ~cluster_idx])
    return intra / (inter + intra)

def modularity(adjacency, clusters):
    degrees = adjacency.sum(axis=0).A1
    m = degrees.sum()
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / m
    return result / m

def convert_torch_npz(adj,features,B,labels):
    features, _ = preprocess_features(features)
    B = torch.FloatTensor(B[np.newaxis])
    adj =normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    #labels = torch.FloatTensor(labels[np.newaxis])
    labels_clus=labels
    #labels_clus=torch.argmax(labels[0],dim=1).detach().cpu().numpy()
    return features,adj,B,labels_clus

def convert_torch_kipf(adj,features,B,labels):
    features, _ = preprocess_features(features)
    B = torch.FloatTensor(B[np.newaxis])
    adj =normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    labels_clus=labels
    labels_clus=torch.argmax(labels[0],dim=1).detach().cpu().numpy()
    return features,adj,B,labels_clus

def load_npz_to_sparse_graph(file_name):    # pylint: disable=missing-function-docstring
    with np.load(open(file_name, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        print(loader.keys())
        adj_matrix = csr_matrix(
                (loader['adj_matrix.data'], loader['adj_matrix.indices'], loader['adj_matrix.indptr']),
                shape=loader['adj_matrix.shape'])

        if 'attr_matrix.data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = csr_matrix(
                    (loader['attr_matrix.data'], loader['attr_matrix.indices'], loader['attr_matrix.indptr']),
                    shape=loader['attr_matrix.shape']).todense()
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            raise Exception('No attributes in the data file', file_name)

        if 'labels.data' in loader:
            # Labels are stored as a CSR matrix
            labels = csr_matrix((loader['labels.data'], loader['labels.indices'],
                                                     loader['labels.indptr']),
                                                    shape=loader['labels.shape'])
            label_mask = labels.nonzero()[0]
            labels = labels.nonzero()[1]
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
            label_mask = np.ones(labels.shape, dtype=np.bool)
        else:
            raise Exception('No labels in the data file', file_name)

    print(labels.shape)
    return adj_matrix, attr_matrix, labels, label_mask
