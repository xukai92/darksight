### Load libraries

import matplotlib
matplotlib.use('Agg')           # for disabling graphical UI
import matplotlib.pyplot as plt
plt.style.use('ggplot')         # for better looking
import matplotlib.cm as cm      # for generating color list
matplotlib.rcParams['xtick.direction'] = 'out'  # let x ticks to behind x-axis
matplotlib.rcParams['ytick.direction'] = 'out'  # let y ticks to left y-axis

# Utility
import csv
import time
import argparse
import os
import sys
import json
import urllib2

# Sci computing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

### End of library loading

### Helper functions

def log_sum_exp_stable_vec(vec):

    max_score = torch.max(vec, keepdim=True)
    max_vec = max_score.expand(vec.size())

    return max_vec + torch.log(torch.sum(torch.exp(vec - max_vec), keepdim=True))

def log_sum_exp_stable_mat(mat):

    n, _ = mat.size()
    max_scores, max_idcs = torch.max(mat, dim=1)
    max_scores = max_scores.view(n, -1)
    max_mat = max_scores.expand(mat.size())

    return max_mat + torch.log(torch.sum(torch.exp(mat - max_mat), dim=1)).view(n, -1)

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

class Knowledge:

    def __init__(self, path_to_logit, data_type="csv", T=1):

        if data_type == "csv":
            logit_f = open(path_to_logit, "r")
            logit_np = np.loadtxt(logit_f)
            logit = torch.from_numpy(logit_np).float()
        elif data_type == "pt":
            logit = torch.load(path_to_logit)
        else:
            print("[Knowledge.__init__] unknown data type of logit: {0}".format(data_type))
            raise
        
        print("[Knowledge.__init__] {0} with size of {1} is loaded".format(type(logit), logit.size()))

        # C - #classes
        # N - #data points
        N, C = logit.size()

        # Convert logit to probability
        logit_div_by_T = logit / T
        p = torch.exp(logit_div_by_T) / torch.sum(torch.exp(logit_div_by_T), 1).view(N, 1).expand(N,C)

        # Log for numerical stability
        log_p = logit_div_by_T - log_sum_exp_stable_mat(logit_div_by_T)
        
        self.N = N
        self.C = C

        self.logit = logit
        self.p = p
        self.log_p = log_p

    def ready(self, use_cuda, gpu_id):

        if use_cuda:
            
            self.logit = self.logit.cuda(gpu_id)
            self.p = self.p.cuda(gpu_id)
            self.log_p = self.log_p.cuda(gpu_id)

        self.logit = Variable(self.logit, requires_grad=False)
        self.p = Variable(self.p, requires_grad=False)
        self.log_p = Variable(self.log_p, requires_grad=False)
