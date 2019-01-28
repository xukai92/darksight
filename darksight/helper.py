import torch


def sym_kl_div(P, Q):
    div_total = 0.5 * torch.sum(P * (torch.log(P) - torch.log(Q))) + \
                0.5 * torch.sum(Q * (torch.log(Q) - torch.log(P)))
    return div_total / P.size(0)


def sym_kl_div_with_log_input(log_P, log_Q):
    div_total = 0.5 * torch.sum(torch.exp(log_P) * (log_P - log_Q)) + \
                0.5 * torch.sum(torch.exp(log_Q) * (log_Q - log_P))
    return div_total / log_P.size(0)
