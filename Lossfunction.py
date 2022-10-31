import torch
import torch.nn as nn
import numpy as np

"""pos_weight1=torch.ones([46])
pos_weight2=torch.ones([46])
pos_weight1=torch.add(pos_weight1,10)

pos_weight = torch.cat((pos_weight1, pos_weight2),0)"""

#reduction= mean for non-overlapping and sum for overlapping.

def loss_modularity_trace(logits, B, nb_community,hid_units,m,modeltype):

    if modeltype=='overlap':
        b_xent = nn.BCEWithLogitsLoss(reduction='sum')
        alpha=0.85
    elif modeltype=='non-overlap':
        b_xent = nn.BCEWithLogitsLoss(reduction='mean')
        alpha=0
    logittrans = logits.transpose(1, 2)
    Module_middle = torch.matmul(logittrans, B)
    Modularity = torch.matmul(Module_middle, logits)
    idx = np.random.permutation(hid_units)
    shuf_fts = Modularity[:, idx, :]
    same_labels = torch.ones(1, hid_units)
    diff_labels = torch.zeros(1, hid_units)
    diff_labels=torch.add(alpha,diff_labels)
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


