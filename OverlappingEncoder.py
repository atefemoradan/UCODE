import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, nb_nodes, hid_units, bias=True):

        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, hid_units, bias=True)
        self.fc1 = nn.Linear(hid_units, out_ft, bias=False)

        self.batch= nn.BatchNorm1d(num_features=nb_nodes)

        self.act = nn.SiLU()
        self.act1=nn.RReLU()


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
        out1 = torch.bmm(adj, seq_fts)
        out1 = self.act(out1)
        out1 = self.batch(out1)

        seq_fts1 = self.fc1(out1)
        seq_fts2 = torch.bmm(adj, seq_fts1)
        seq_fts2 = self.batch(seq_fts2)
        seq_fts2 = self.act1(seq_fts2)



        return seq_fts2

