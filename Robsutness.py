import torch
import os

dataset='cora'
path=os.path.dirname(os.path.abspath(__file__))+'/venv/'


model = torch.load(path+'model/'+dataset+'_model.pt')
model.eval()