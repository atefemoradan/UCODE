
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
import torch.nn as nn

path=os.path.dirname(os.path.abspath(__file__))+'/venv/'
# From Dmon Model
def _compute_counts(y_true, y_pred):  #
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

def load_npz_to_sparse_graph(file_name):  # pylint: disable=missing-function-docstring
  with np.load(open(file_name, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)
    adj_matrix = csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    if 'attr_data' in loader:
      # Attributes are stored as a sparse CSR matrix
      attr_matrix = csr_matrix(
          (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
          shape=loader['attr_shape']).todense()
    elif 'attr_matrix' in loader:
      # Attributes are stored as a (dense) np.ndarray
      attr_matrix = loader['attr_matrix']
    else:
      raise Exception('No attributes in the data file', file_name)

    if 'labels_data' in loader:
      # Labels are stored as a CSR matrix
      labels = csr_matrix((loader['labels_data'], loader['labels_indices'],
                           loader['labels_indptr']),
                          shape=loader['labels_shape'])
      label_mask = labels.nonzero()[0]
      labels = labels.nonzero()[1]
    elif 'labels' in loader:
      # Labels are stored as a numpy array
      labels = loader['labels']
      label_mask = np.ones(labels.shape, dtype=np.bool)
    else:
      raise Exception('No labels in the data file', file_name)

  return adj_matrix, attr_matrix, labels, label_mask




############# This code is taken from Ovelapping-Günnemann ############

import scipy.sparse as sp
from typing import Union

def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     cuda: bool = True,
                     ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:
  """Convert a scipy sparse matrix to a torch sparse tensor.

  Args:
      matrix: Sparse matrix to convert.
      cuda: Whether to move the resulting tensor to GPU.

  Returns:
      sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

  """
  if sp.issparse(matrix):
    coo = matrix.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
  elif torch.is_tensor(matrix):
    row, col = matrix.nonzero().t()
    indices = torch.stack([row, col])
    values = matrix[row, col]
    shape = torch.Size(matrix.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
  else:
    raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
  if cuda:
    sparse_tensor = sparse_tensor.cuda()
  return sparse_tensor.coalesce()

# Loading overlapping dataset
def load_overlapping_dataset(file_name):
  """Load a graph from a Numpy binary file.

  Parameters
  ----------
  file_name : str
      Name of the file to load.

  Returns
  -------
  graph : dict
      Dictionary that contains:
          * 'A' : The adjacency matrix in sparse matrix format
          * 'X' : The attribute matrix in sparse matrix format
          * 'Z' : The community labels in sparse matrix format
          * Further dictionaries mapping node, class and attribute IDs

  """
  if not file_name.endswith('.npz'):
    file_name += '.npz'
  with np.load(file_name, allow_pickle=True) as loader:
    loader = dict(loader)
    A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                       loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])

    if 'attr_matrix.data' in loader.keys():
      X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                         loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
    else:
      X = None

    Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                       loader['labels.indptr']), shape=loader['labels.shape'])

    # Remove self-loops
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()

    # Convert label matrix to numpy
    if sp.issparse(Z):
      Z = Z.toarray().astype(np.float32)

    graph = {
      'A': A,
      'X': X,
      'Z': Z
    }

    node_names = loader.get('node_names')
    if node_names is not None:
      node_names = node_names.tolist()
      graph['node_names'] = node_names

    attr_names = loader.get('attr_names')
    if attr_names is not None:
      attr_names = attr_names.tolist()
      graph['attr_names'] = attr_names

    class_names = loader.get('class_names')
    if class_names is not None:
      class_names = class_names.tolist()
      graph['class_names'] = class_names

    return graph


__all__ = [
    'GCN',
    'GraphConvolution',
]


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)


class GraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([GraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]

# ORECAL AND ONMI for the overlapping community detection

def overlap_nmi(X, Y):
  if not ((X == 0) | (X == 1)).all():
    raise ValueError("X should be a binary matrix")
  if not ((Y == 0) | (Y == 1)).all():
    raise ValueError("Y should be a binary matrix")

  # if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
  #     warnings.warn("It seems that you forgot to transpose the F matrix")
  X = X.T
  Y = Y.T

  def cmp(x, y):
    """Compare two binary vectors."""
    a = (1 - x).dot(1 - y)
    d = x.dot(y)
    c = (1 - y).dot(x)
    b = (1 - x).dot(y)
    return a, b, c, d

  def h(w, n):
    """Compute contribution of a single term to the entropy."""
    if w == 0:
      return 0
    else:
      return -w * np.log2(w / n)

  def H(x, y):
    """Compute conditional entropy between two vectors."""
    a, b, c, d = cmp(x, y)
    n = len(x)
    if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
      return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
    else:
      return h(c + d, n) + h(a + b, n)

  def H_uncond(X):
    """Compute unconditional entropy of a single binary matrix."""
    return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

  def H_cond(X, Y):
    """Compute conditional entropy between two binary matrices."""
    m, n = X.shape[0], Y.shape[0]
    scores = np.zeros([m, n])
    for i in range(m):
      for j in range(n):
        scores[i, j] = H(X[i], Y[j])
    return scores.min(axis=1).sum()

  if X.shape[1] != Y.shape[1]:
    raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
  H_X = H_uncond(X)
  H_Y = H_uncond(Y)
  I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
  return I_XY / max(H_X, H_Y)


def ORecall(thresh,l,c,Z_gt):
  #l=cluster_assignment(thresh)
  ground_truth=[set() for i in range(c)]
  dict_assign=dict()
  for i in range(len(l[0])):
    dict_assign[i]=[]
  for i in range(len(l)):
    for j in range(len(l[i])):
      if l[i][j]==True:
        dict_assign[j].append(i)
  for i in range(len(Z_gt)):
    for j in range(len(Z_gt[i])):
      if Z_gt[i][j]==1:
        ground_truth[j].add(i)
  detected_communites=detect_community_by_recall(dict_assign,ground_truth,c)
  average_recall = cal_avg_recall(detected_communites,c)
  return average_recall

def detect_community_by_recall(assigned_communties,act_comm,c):
  all={}
  for a in range(0,c): # For each community in ground truth
    each = []
    recall = -1
    best_match_comm = -1
    for p in range(0,c): # For each in assignec coomunities (As communities may shift positions)
      if p in assigned_communties.keys():
        if len(act_comm[a].intersection(assigned_communties[p]))/len(act_comm[a]) > recall:
          recall = len(act_comm[a].intersection(assigned_communties[p]))/len(act_comm[a]) # as per recall formula
          best_match_comm = p
    each.append(best_match_comm) #appenindg the best matched coounity from assigned
    each.append(recall) # and its recall score
    all[a] = each
    assigned_communties.pop(best_match_comm) # removing the best matched detected community
  return all # retufing thr final dictionary with gccomm:[deteced comm, recall]

def cal_avg_recall(detected_communites,c):
  avg_recall=0
  for i in range(0,c):
    avg_recall+=detected_communites[i][1]
  return avg_recall/c


def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     cuda: bool = False,
                     ) -> Union[torch.sparse.FloatTensor, torch.sparse.FloatTensor]:
  """Convert a scipy sparse matrix to a torch sparse tensor.

  Args:
      matrix: Sparse matrix to convert.
      cuda: Whether to move the resulting tensor to GPU.

  Returns:
      sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

  """
  if sp.issparse(matrix):
    coo = matrix.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
  elif torch.is_tensor(matrix):
    row, col = matrix.nonzero().t()
    indices = torch.stack([row, col])
    values = matrix[row, col]
    shape = torch.Size(matrix.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
  else:
    raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
  if cuda:
    sparse_tensor = sparse_tensor
  return sparse_tensor.coalesce()


def normalize_overlap_adj(adj: sp.csr_matrix):
  """Normalize adjacency matrix and convert it to a sparse tensor."""
  if sp.isspmatrix(adj):
    adj = adj.tolil()
    adj.setdiag(1)
    adj = adj.tocsr()
    deg = np.ravel(adj.sum(1))
    deg_sqrt_inv = 1 / np.sqrt(deg)
    adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
  elif torch.is_tensor(adj):
    deg = adj.sum(1)
    deg_sqrt_inv = 1 / torch.sqrt(deg)
    adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
  return to_sparse_tensor(adj_norm)

