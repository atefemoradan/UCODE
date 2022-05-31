import math
import os
import pickle as pkl

from UCODEncoder import GCN
import utils
from Lossfunction import loss_modularity_trace

from tqdm.auto import tqdm
import networkx as nx
from networkx.generators.degree_seq import expected_degree_graph

from sklearn.cluster import KMeans
from umap import UMAP
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy import sparse
from sklearn import metrics

def datapreprocessing(data_type, path, dataset, exist_B=1):
    if data_type == 'kipf':
        adj, features, labels = utils.load_data(dataset)
        with open(path + "dataset/ind." + str(dataset) + ".graph", 'rb') as f:
            Graph = pkl.load(f, encoding='latin1')
        network = nx.Graph(Graph)
        label_mask = np.ones_like(labels)
    elif data_type == "npz":
        adj, features, labels, label_mask = utils.load_npz_to_sparse_graph(path + 'dataset/' + dataset +'.npz')
        network = nx.from_scipy_sparse_matrix(adj)
        features = sparse.csr_matrix(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    m = len(network.edges)

    if exist_B == 0:
        B=utils.get_B(network)
        np.save(path+'dataset/Modularity-'+dataset, B)
    else:
        B = np.load(path+'dataset/Modularity-'+dataset+'.npy')

    if data_type=="kipf":
        features, adj, B, true_labels = utils.convert_torch_kipf(adj, features, B, labels)
    elif data_type=="npz":
        features, adj, B, true_labels = utils.convert_torch_npz(adj, features, B, labels)

    return features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj, m

def train(
    dataset_dict,
    hid_units,
    data_type,
    path,
    epochs=500,
    num_experiments=5,
    _overlap_threshold=-1
):
    for data_name, n_communities in tqdm(dataset_dict.items()):
        print(data_name)

        preprocessed_data = datapreprocessing(data_type, path, data_name)
        features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj, m = preprocessed_data

        nmi_list = np.zeros(num_experiments)

        # these metrics are only used in non-overlapping setting
        if data_type == 'kipf':
            kmeans_nmi_list = np.zeros(num_experiments)
            cond_list, kmeans_cond_list = np.zeros(num_experiments), np.zeros(num_experiments)
            modularity_list, kmeans_modularity_list = np.zeros(num_experiments), np.zeros(num_experiments)
            Fscore_list, kmeans_Fscore_list = np.zeros(num_experiments), np.zeros(num_experiments)

        for i in tqdm(range(num_experiments)):
            model = GCN(ft_size, hid_units, nb_nodes)
            optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
            if torch.cuda.is_available():
                model.cuda()
                features = features.cuda()
                B = B.cuda()
                adj = adj.cuda()

            # Training loop
            for epoch in range(epochs):
                model.train()
                optimiser.zero_grad()
                logits = model(features, adj)
                loss = loss_modularity_trace(logits, B, n_communities, hid_units, m)
                loss.backward()
                optimiser.step()

            # Get predictions at the end of training
            model.eval()
            logits = model(features, adj)
            # logits = torch.exp(logits)
            cpu_logits = logits.view(nb_nodes, hid_units).detach().cpu().numpy()
            preds = torch.argmax(logits[0], dim=1).detach().cpu().numpy()

            # Fit kmeans to the output space
            kmeans_model = KMeans(
                init='k-means++',
                n_clusters=n_communities,
                random_state=0
            ).fit(cpu_logits)
            kmeans_preds = kmeans_model.labels_

            # Calculate NMI
            if data_type == 'npz':
                if _overlap_threshold < 0:
                    overlap_threshold = np.mean(cpu_logits, axis=-1, keepdims=True)
                else:
                    overlap_threshold = _overlap_threshold

                pred_mask = (cpu_logits > overlap_threshold).astype(np.int32)
                nmi = utils.overlap_nmi(label_mask, pred_mask)
            else:
                nmi = normalized_mutual_info_score(true_labels, preds)
                kmeans_nmi = normalized_mutual_info_score(true_labels, kmeans_preds)
                kmeans_nmi_list[i] = kmeans_nmi
            nmi_list[i] = nmi

            # Calculate other non-overlapping metrics
            if data_type == 'kipf':
                cpu_adj = adj[0].detach().cpu().numpy() # adj[0] to get rid of the batch dimension
                cond_list[i] = utils.conductance(cpu_adj, preds)
                kmeans_cond_list[i] = utils.conductance(cpu_adj, kmeans_preds)
                modularity_list[i] = utils.modularity(cpu_adj, preds)
                kmeans_modularity_list[i] = utils.modularity(cpu_adj, kmeans_preds)
                p = utils.precision(true_labels, preds)
                r = utils.recall(true_labels, preds)
                Fscore_list[i] = 2 * (r * p) / (r + p)

                p = utils.precision(true_labels, kmeans_preds)
                r = utils.recall(true_labels, kmeans_preds)
                kmeans_Fscore_list[i] = 2 * (r * p) / (r + p)


        if data_type == 'npz':
            print('NMI:', np.mean(nmi_list), np.std(nmi_list))
        else:
            print('Baseline scores:')
            print('NMI:', np.mean(nmi_list), np.std(nmi_list))
            print('Fscore:', np.mean(Fscore_list), np.std(Fscore_list))
            print('Conductance:', np.mean(cond_list), np.std(cond_list))
            print('Modularity:', np.mean(modularity_list), np.std(modularity_list))

            print('Kmeans scores:')
            print('NMI:', np.mean(kmeans_nmi_list), np.std(kmeans_nmi_list))
            print('Fscore:', np.mean(kmeans_Fscore_list), np.std(kmeans_Fscore_list))
            print('Conductance:', np.mean(kmeans_cond_list), np.std(kmeans_cond_list))
            print('Modularity:', np.mean(kmeans_modularity_list), np.std(kmeans_modularity_list))

def run_nonoverlapping():
    path=os.path.dirname(os.path.abspath(__file__))+'/venv/'
    dataset_dict = {
        'cora': 7,
        'pubmed': 3,
        'citeseer': 6,
    }
    hid_units=16
    epochs=500
    data_type='kipf'
    train(dataset_dict, hid_units, data_type, path, epochs=epochs)

def run_overlapping():
    path=os.path.dirname(os.path.abspath(__file__))+'/venv/'
    dataset_dict = {
        'fb_348': 14,
        'fb_414': 7,
        'fb_686': 14,
        'fb_698': 13,
        'fb_1684': 17,
        'fb_1912': 46
    }
    hid_units=16
    epochs=500
    data_type='npz'
    train(dataset_dict, hid_units, data_type, path, epochs=epochs)

if __name__ == '__main__':
    # run_nonoverlapping()
    run_overlapping()
