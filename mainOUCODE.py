import nocd
import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

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
from UCODEncoder import GCN

if torch.cuda.is_available():
    device='gpu'
else:
    device='cpu'


path= os.path.dirname(os.path.abspath(__file__))
existB=1




def datapreprocessing(overlapmodel,path, dataset, exist_B=1):
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
  if existB:
    B = com.get_B(network)
  else:
    B = np.load(path + '/dataset/Overlappingdatasets/' + dataset)
  B = torch.FloatTensor(B)
  m = 2 * len(network.edges)
  return x_norm,A,B,Z_gt,m


def train(dataset_dict,
    hid_units,
    path,
    overlapmodel,
    epochs=500,
    num_experiments=10,
):
  print(overlapmodel)
  for data_name, n_communities in tqdm(dataset_dict.items()):
    print(data_name)

    preprocessed_data = datapreprocessing(overlapmodel,path, data_name, exist_B=1)
    x_norm_i, A, B, Z_gt, m = preprocessed_data
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
        loss = loss_modularity_trace(Z, B, c,c,m)
        #print('Loss:', loss.item())
        #logits = Z.cpu().detach().numpy()
        #loss += nocd.utils.l2_reg_loss(gnn, scale=weight_decay)
        loss.backward()
        optimiser.step()
      logits = torch.exp(Z)
      logits = logits.cpu().detach().numpy()
      thresh =np.mean(np.mean(logits, axis=-1, keepdims=True))
      preds = logits > thresh
      preds=np.squeeze(preds, axis=0)
      nmi=com.overlap_nmi(Z_gt,preds)
      print(nmi)
      recall=com.ORecall(thresh,preds,c,Z_gt)
      recall_list[i]=recall
      nmi_list[i] = nmi

    print('The average of ONMI is='+str(np.mean(nmi_list)))
    print('The average of ORecall is=' + str(np.mean(recall_list)))



def run_overlapping():
  path=os.path.dirname(os.path.abspath(__file__))
  dataset_dict = {
      'fb_348': 14

  }
  overlapmodel='UCODE-G'
  hid_units=128
  epochs=300
  train(dataset_dict, hid_units, path,overlapmodel, epochs)


if __name__ == '__main__':
  run_overlapping()

# Fb_348, hid_units=128,epoch=300
# Fb_414, hid_units=128,epoch=300

