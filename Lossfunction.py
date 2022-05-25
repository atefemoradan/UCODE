import torch
import torch.nn as nn
import numpy as np

b_xent = nn.BCEWithLogitsLoss()


def loss_modularity_trace(logits, B, nb_community,hid_units,m):
    logittrans = logits.transpose(1, 2)
    Module_middle = torch.matmul(logittrans, B)
    Modularity = torch.matmul(Module_middle, logits)
    idx = np.random.permutation(hid_units)
    shuf_fts = Modularity[:, idx, :]
    lbl_1 = torch.ones(1, hid_units)
    lbl_2 = torch.zeros(1, hid_units)
    lbl = torch.cat((lbl_2, lbl_1), 1)
    diag1 = torch.div(torch.diag(Modularity[0], 0), m)
    diag2 = torch.div(torch.diag(shuf_fts[0], 0), m)
    #print("diag1 :" + str(torch.sum(diag1)))
    #print("diag2 :" + str(torch.sum(diag2)))
    input = torch.cat((diag2, diag1), 0)
    input = input.unsqueeze(0)
    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
        input = input.cuda()

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    loss = b_xent(input, lbl)
    loss.requires_grad_(True)
    return loss


"""def loss_modularity_trace1(logits, B, nb_community,hid_units,m):

    logittrans = logits.transpose(1, 2)
    Module_middle = torch.matmul(logittrans, B)
    Modularity = torch.matmul(Module_middle, logits)

    idx = np.random.permutation(hid_units)
    shuf_fts = Modularity[:, idx, :]
    # idx = np.random.permutation(Modularity)


    diag1 = torch.diag(Modularity[0], 0)
    diag2 = torch.diag(shuf_fts[0], 0)
    lbl_1 = torch.ones(1, hid_units)
    lbl_2 = torch.zeros(1, hid_units)
    lbl = torch.cat((lbl_1, lbl_2), 1)
    # utput1= Scoring_function(diag1, logits)
    # output2= Scoring_function(diag2, logits)
    input = torch.cat((diag1, diag2), 0)
    input = input.unsqueeze(0)

    if torch.cuda.is_available():
        lbl = lbl.cuda()
        input = input.cuda()
    loss =-b_xent(input, lbl)
    # diag1=torch.FloatTensor(diag1[np.newaxis])
    # diag2=torch.FloatTensor(diag2[np.newaxis])
    input = torch.cat((diag2, diag1), 0)
    input = input.unsqueeze(0)
    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()

        lbl = lbl.cuda()
        input = input.cuda()
    # logits, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

    loss = b_xent(input, lbl)
    return loss"""
