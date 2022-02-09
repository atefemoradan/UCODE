import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, nb_nodes, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, 512, bias=True)
        self.fc1 = nn.Linear(512, out_ft, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.batch = nn.BatchNorm1d(num_features=nb_nodes)
        self.act = nn.SELU()
        self.act1 = nn.LogSoftmax(dim=2)
        self.act2 = nn.ReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data, gain=2.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out1 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out1 = torch.bmm(adj, seq_fts)
        out1 = self.act(out1)
        out1 = self.batch(out1)
        seq_fts1 = self.fc1(out1)
        if sparse:
            seq_fts2 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts1, 0)), 0)
        else:
            seq_fts2 = torch.bmm(adj, seq_fts1)
        seq_fts2 = self.batch(seq_fts2)
        seq_fts2 = self.act2(seq_fts2)
        #out2 = self.act1(torch.log(seq_fts2))
        out2 = self.act1(seq_fts2)
        return out2

##
"""class GCN1(nn.Module):
    def __init__(self, in_ft, out_ft,nb_nodes, bias=True):
        super(GCN1, self).__init__()
        self.fc = nn.Linear(in_ft,512, bias=True)
        self.fc1 = nn.Linear(512,out_ft, bias=False)
        self.dropout= nn.Dropout(p=0.3)
        #self.fc2 = nn.Linear(1024,7, bias=False)
        self.batch = nn.BatchNorm1d(num_features=nb_nodes)
        #self.batch1 = nn.BatchNorm1d(num_features=nb_nodes)
        #self.batch2 = nn.BatchNorm1d(num_features=nb_nodes)
        self.act = nn.ReLU()
        self.act1=nn.LogSoftmax(dim=2)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data,gain=2.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
      #first layer    
        seq_fts = self.fc(seq)
        if sparse:
          out1 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
          out1 = torch.bmm(adj, seq_fts)

        out1=self.act(out1)
        out1=self.batch(out1)
        out1=self.dropout(out1)
        seq_fts1 = self.fc1(out1)
        if sparse:
            out2 = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts1, 0)), 0)
        else:
            out2 = torch.bmm(adj, seq_fts1)
        out2=self.batch(out2)
        out2=self.act(out2)


        return out2"""