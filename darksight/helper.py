import torch


def sym_kl_div(P, Q):
    return _sym_kl_div(torch.log(P) - torch.log(Q), P, Q)


def sym_kl_div_with_log_input(log_P, log_Q):
    return _sym_kl_div(log_P - log_Q, torch.exp(log_P), torch.exp(log_Q))
  
  
def _sym_kl_div(logdiff, P, Q):
    div_total = 0.5 * torch.sum(P * ( logdiff)) + \
                0.5 * torch.sum(Q * (-logdiff))
    return div_total / logdiff.size(0)
