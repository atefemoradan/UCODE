import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm
import omega_index_py3
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import utils as com
import pandas as pd
from sklearn.preprocessing import normalize
import networkx as nx
from Lossfunction import loss_modularity_trace
import pickle
import sys
import scipy.sparse as sp
import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import os
from OverlappingEncoder import GCN
import math
if torch.cuda.is_available():
    device='gpu'
else:
    device='cpu'


path= os.path.dirname(os.path.abspath(__file__))
exist_B=1




def datapreprocessing(overlapmodel,path, dataset, exist_B):
  loader = com.load_overlapping_dataset(path + '/dataset/Overlappingdatasets/' + dataset)
  A, X, Z_gt = loader['A'], loader['X'], loader['Z']
  N, K = Z_gt.shape

  if overlapmodel=='UCODE-G':
    x_norm = normalize(X)  # node features
    x_norm = normalize(A)  # adjacency matrix
    x_norm = sp.hstack([normalize(X), normalize(A)])  # concatenate A and X
    x_norm = com.to_sparse_tensor(x_norm)
  elif overlapmodel=='UCODE-X':
    x_norm = normalize(X)  # node features
    x_norm = com.to_sparse_tensor(x_norm)
  """calculate-B-Matrix"""
  network = nx.from_scipy_sparse_matrix(A)
  #nx.write_edgelist(network, path+"/edgelist-overlap/"+dataset, comments='#',data=False)
  if exist_B == 0:
    B = com.get_B(network)
    np.save(path + '/dataset/Overlappingdatasets/Modularity-' + dataset, B)
  else:
    B = np.load(path + '/dataset/Overlappingdatasets/Modularity-' + dataset + '.npy')
    print(B.shape)
  B = torch.FloatTensor(B)
  m = 2 * len(network.edges)
  return x_norm,A,B,Z_gt,m


def train(dataset_dict,
    hid_units,
    path,
    overlapmodel,
    epochs=500,
    num_experiments=9,
):
  sigmoidlogit = nn.Sigmoid()
  print(overlapmodel)
  for data_name, n_communities in tqdm(dataset_dict.items()):
    print(data_name)

    preprocessed_data = datapreprocessing(overlapmodel,path, data_name, exist_B)
    x_norm_i, A, B, Z_gt, m = preprocessed_data
    #edge=sum(sum(A.toarray())/2)
    #node=B.shape[0]
    #epsilon=2*edge/(node*(node-1))
    #thresh=math.sqrt(-math.log((1-epsilon),10))
    #print("threshold:",thresh)
    c = Z_gt.shape[1]
    adj_norm = com.normalize_overlap_adj(A)
    x_norm = x_norm_i.to_dense()
    adj_norm = adj_norm.to_dense()
    x_norm = torch.unsqueeze(x_norm, dim=0)
    adj_norm = torch.unsqueeze(adj_norm, dim=0)
    nmi_list = np.zeros(num_experiments)
    recall_list= np.zeros(num_experiments)
    for i in tqdm(range(num_experiments)):
      model = None
      model = GCN(x_norm_i.shape[1],c , x_norm_i.shape[0],hid_units)
      optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
      for epoch in range(epochs):
        # Training step
        model.train()
        optimiser.zero_grad()
        Z = model(x_norm, adj_norm)
        loss = loss_modularity_trace(Z, B, c,c,m,'overlap')
        #print('Loss:', loss.item())
        #logits = Z.cpu().detach().numpy()
        #loss += nocd.utils.l2_reg_loss(gnn, scale=weight_decay)
        loss.backward()
        optimiser.step()
      Z = sigmoidlogit(Z)
      #Z = torch.exp(Z)
      logits = Z.cpu().detach().numpy()
      thresh =np.mean(np.mean(logits, axis=-1, keepdims=True))
      thresh=0.9
      preds = logits > thresh
      preds=np.squeeze(preds, axis=0)
      #preds={preds[i]: i for i in range(len(preds))}
      nmi=com.overlap_nmi(Z_gt,preds)
      print('NMI :'+ str(nmi))
      recall=com.ORecall(thresh,preds,c,Z_gt)
      recall_list[i]=recall
      nmi_list[i] = nmi

    print('The average of ONMI is='+str(np.mean(nmi_list)))
    print('The average of ORecall is=' + str(np.mean(recall_list)))
    print(np.std(nmi_list))

def run_overlapping():
  path=os.path.dirname(os.path.abspath(__file__))
  """'fb_348':14,
  'fb_414':7,
  'fb_686':14,
  'fb_698':13,
  'fb_1684':17,
  'fb_1912':46
  'mag_eng':16"""
  dataset_dict = {
    'fb_698':13

  }
  overlapmodel='UCODE-X'
  hid_units=128
  epochs=500
  train(dataset_dict, hid_units, path,overlapmodel, epochs)


if __name__ == '__main__':
  run_overlapping()

# Fb_348, hid_units=128,epoch=300
# Fb_414, hid_units=128,epoch=300

