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

# Sci computing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from helper import *

### End of library loading

class Distribution:

    def pdf(self):

        raise NotImplementedError()

    def logpdf(self):

        raise NotImplementedError()

    def cuda(self, gpu_id):

        for i in range(len(self.params)):

            self.params[i] = self.params[i].cuda(gpu_id)

    def var(self):

        for i in range(len(self.params)):

            self.params[i] = Variable(self.params[i], requires_grad=True)

    def ready(self, use_cuda, gpu_id):

        if use_cuda:
            
            self.cuda(gpu_id)

        self.var()
        
class NormalFamily(Distribution):

    def __init__(self, C, D):

        mu = torch.randn([C, D]) * np.sqrt(C)
        H = torch.zeros([C, D, D])
        for c in range(C):
            H[c,:,:] = torch.eye(D, D) / np.sqrt(np.log(C))

        self.params = [mu, H]
        self.C = C
        self.D = D

    def pre(self, y):

        n, _ = y.size()
        C, D = self.C, self.D

        mu, H = self.params
        
        det_H = H[:,0,0] * H[:,1,1] - H[:,0,1] * H[:,1,0]
        y_expd = y.view([n, 1, D]).expand([n, C, D])
        mu_expd = mu.view([1, C, D]).expand([n, C, D])
        y_mu_diff = y_expd - mu_expd

        return det_H, y_mu_diff, H, n, C, D

class Gaussian(NormalFamily):

    def logpdf(self, y):

        det_H, y_mu_diff, H, n, C, D = self.pre(y)

        lp = (0.5 * torch.log(torch.abs(det_H)) - np.log(2 * np.pi) - \
              0.5 * torch.matmul(torch.matmul(y_mu_diff.view(n, C, 1, D), H), 
                                 y_mu_diff.view(n, C, D, 1)).view(n, C))

        return lp

    def pdf(self, y):

        det_H, y_mu_diff, H, n, C, D = self.pre(y)

        p = (torch.sqrt(torch.abs(det_H)) / (2 * np.pi) * \
             torch.exp(-0.5 * torch.matmul(torch.matmul(y_mu_diff.view(n, C, 1, D), H), 
                                           y_mu_diff.view(n, C, D, 1))).view(n, C))

        return p

class Student(NormalFamily):

    def logpdf(self, y):

        det_H, y_mu_diff, H, n, C, D = self.pre(y)

        lp = (0.5 * torch.log(torch.abs(det_H)) - np.log(2 * np.pi) - \
              2.0 * torch.log(1.0 + \
                              0.5 * torch.matmul(torch.matmul(y_mu_diff.view(n, C, 1, D), H), 
                                                 y_mu_diff.view(n, C, D, 1))).view(n, C))

        return lp

    def pdf(self, y):

        det_H, y_mu_diff, H, n, C, D = self.pre(y)

        p = (torch.sqrt(torch.abs(det_H)) / (2 * np.pi) / \
             (1 + 0.5 * torch.matmul(torch.matmul(y_mu_diff.view(n, C, 1, D), H), 
                                     y_mu_diff.view(n, C, D, 1))).view(n, C)**2)

        return p

class Softmax(Distribution):

    def __init__(self, C):

        W = torch.ones([C]) * np.log(10)

        self.params = [W]

    def logpdf(self):

        W, = self.params

        lp = W - log_sum_exp_stable_vec(W)

        return lp

    def pdf(self):

        W, = self.params

        p = torch.exp(W) / torch.sum(torch.exp(W))

        return p
