from UCODEncoder import GCN
import utils as com
from sklearn.cluster import KMeans
import pickle as pkl
import networkx as nx
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from Lossfunction import loss_modularity_trace
from scipy import sparse
import os
from sklearn import metrics
from utils import precision
from utils import recall

from networkx.generators.degree_seq import expected_degree_graph

if torch.cuda.is_available():
        device='gpu'
else:
        device='cpu'

"""Parameters explanation:
type --> kipf or npz
If the modularity matrix (B) is saved in the system existB would be equal to 1"""

"""Set the path of the dataset folder"""
path=os.path.dirname(os.path.abspath(__file__))+'/venv/'
type='npz'
dataset ='cora'
ncommunity=7
hid_units=16
existB=1
epochs=500

def datapreprocessing():
    if type=='kipf':
        adj, features, labels = com.load_data(dataset)
        with open(path + "dataset/ind." + str(dataset) + ".graph", 'rb') as f:
            Graph = pkl.load(f, encoding='latin1')
        network = nx.Graph(Graph)
    elif type=="npz":
        adj, features, labels, label_mask = com.load_npz_to_sparse_graph(path + 'dataset/' + dataset +'.npz')
        network = nx.from_scipy_sparse_matrix(adj)
        features = sparse.csr_matrix(features)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    m=len(network.edges)
    if existB==0:
        B=com.get_B(network)
        print(B.shape)
        adj_metric=adj
        np.save(path+'dataset/Modularity-'+dataset+'npy',B)
    else:
        B = np.load(path+'dataset/Modularity-'+dataset+'.npy')
        adj_metric = adj
    if type=="npz":
        features,adj,B,labels_clus=com.convert_torch_npz( adj,features,B,labels)
    elif type=="kipf":
        features, adj, B, labels_clus = com.convert_torch_kipf(adj, features, B, labels)
    return features,adj,B,labels_clus,nb_nodes,ft_size,adj_metric,m

def training():
    features,adj,B,labels_clus,nb_nodes,ft_size,adj_metric,m=datapreprocessing()
    nmilist,modularitylist,conductancelist,cslist,precisionlist,recallist,Fscorelist,nmikmean=[],[],[],[],[],[],[],[]
    for i in range(10):
        model = None
        model = GCN(ft_size, hid_units, nb_nodes)
        #lr=0.0001, weight_decay=0.000001
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        if torch.cuda.is_available():
            print('Using CUDA')
            model.cuda()
            features = features.cuda()
            B = B.cuda()
            adj = adj.cuda()
        for epoch in range(epochs):
            print(epoch)
            model.train()
            optimiser.zero_grad()
            logits = model(features, adj)
            loss = loss_modularity_trace(logits, B, ncommunity,hid_units,m)
            logits = torch.exp(logits)
            labels = torch.argmax(logits[0], dim=1).detach().cpu().numpy()
            print(labels.shape)
            print(labels_clus.shape)
            quit()
            nmi=normalized_mutual_info_score(labels_clus, labels)
            print('Loss:', loss.item())
            print('NMI:,',nmi)
            loss.backward()
            optimiser.step()
        logits = torch.exp(logits)
        labels = torch.argmax(logits [0], dim=1).detach().cpu().numpy()
        nmilist.append(normalized_mutual_info_score(labels_clus, labels))
        torch.save(model, path+'model/'+dataset+'_model.pt')
        modularitylist.append(com.modularity(adj_metric,labels))
        conductancelist.append(com.conductance(adj_metric,labels))
        cslist.append(metrics.completeness_score(labels_clus,labels))
        p=precision(labels_clus,labels)
        r=recall(labels_clus,labels)
        precisionlist.append(p)
        recallist.append(r)
        Fscorelist.append(2*((r*p)/(r+p)))
        embeds1 = logits.view(nb_nodes, hid_units).detach().cpu().numpy()
        #kmeans1 = KMeans(init='k-means++', n_clusters=ncommunity, random_state=0).fit(embeds1)
        #nmikmean.append (normalized_mutual_info_score(labels_clus, kmeans1.labels_))
        print(nmilist)
        np.save(path + '/dataset/coductance'+dataset+'.npy', conductancelist)
        np.save(path + '/dataset/fscore+'+dataset+'.npy', Fscorelist)
        np.save(path + '/dataset/modularity'+dataset+'.npy', modularitylist)
        np.save(path + '/dataset/nmi'+dataset+'.npy', nmikmean)

if __name__ == '__main__':
    training()
