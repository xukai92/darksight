### Load libraries

import torch
from torch.autograd import Variable

### End of library loading

### Helper functions

def log_sum_exp_stable_vec(vec):

    max_score, max_idx = torch.max(vec, 0, keepdim=True)
    max_vec = max_score.expand(vec.size())

    return max_vec + torch.log(torch.sum(torch.exp(vec - max_vec), 0, keepdim=True))

def log_sum_exp_stable_mat(mat):

    n, _ = mat.size()
    max_scores, max_idcs = torch.max(mat, 1)
    max_scores = max_scores.view(n, -1)
    max_mat = max_scores.expand(mat.size())

    return max_mat + torch.log(torch.sum(torch.exp(mat - max_mat), 1)).view(n, -1)

def sym_kl_div(P, Q):

    n, _ = P.size()
    div_total = 0.5 * torch.sum(P * (torch.log(P) - torch.log(Q))) + \
                0.5 * torch.sum(Q * (torch.log(Q) - torch.log(P)))
    return div_total / n

def sym_kl_div_with_log_input(log_P, log_Q):

    n, _ = log_P.size()
    div_total = 0.5 * torch.sum(torch.exp(log_P) * (log_P - log_Q)) + \
        0.5 * torch.sum(torch.exp(log_Q) * (log_Q - log_P))
    return div_total / n

def pt2np(pt):

    if type(pt) == torch.autograd.variable.Variable:

        return pt.cpu().data.numpy()

    else:

        return pt.cpu().numpy()