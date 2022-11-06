import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, nb_nodes, hidden_size):
        super(GCN, self).__init__()
        self.fc_1 = nn.Linear(in_ft, hidden_size, bias=False)
        self.fc_2 = nn.Linear(hidden_size, out_ft, bias=False)

        self.batch = nn.BatchNorm1d(num_features=nb_nodes)


        self.act_1=nn.SiLU()
        self.act_2=nn.RReLU()


        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data, gain=2.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        h = self.fc_1(seq)
        h = torch.bmm(adj, h)
        h = self.act_1(h)
        h = self.batch(h)

        h = self.fc_2(h)
        h = torch.bmm(adj, h)
        h= self.batch(h)
        h=self.act_2(h)

        return h
