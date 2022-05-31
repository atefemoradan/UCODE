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

    same_labels = torch.ones(1, hid_units)
    diff_labels = torch.zeros(1, hid_units)
    loss_labels = torch.cat((diff_labels, same_labels), 1)

    same_diag = torch.div(torch.diag(Modularity[0], 0), m)
    diff_diag = torch.div(torch.diag(shuf_fts[0], 0), m)
    similarity = torch.cat((diff_diag, same_diag), 0)
    similarity = similarity.unsqueeze(0)
    if torch.cuda.is_available():
        loss_labels = loss_labels.cuda()
        similarity = similarity.cuda()

    loss = b_xent(similarity, loss_labels)
    loss.requires_grad_(True)

    return loss
