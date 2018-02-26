### Load libraries

import matplotlib
matplotlib.use('Agg')           # for disabling graphical UI
import matplotlib.pyplot as plt
plt.style.use('ggplot')         # for better looking
import matplotlib.cm as cm      # for generating color list
matplotlib.rcParams['xtick.direction'] = 'out'  # let x ticks to behind x-axis
matplotlib.rcParams['ytick.direction'] = 'out'  # let y ticks to left y-axis
from matplotlib.backends.backend_pdf import PdfPages

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

class DarkSight:

    def __init__(self, klg, clf):
        '''
        @input
            klg         :   knowledge
            clf         :   low-dimensional classifier
        '''

        self.y = torch.zeros([klg.N, clf.like_dist.D])
        self.klg = klg
        self.clf = clf

    def align(self):

        mu = self.clf.like_dist.params[0]
        _, max_idcs = torch.max(self.klg.p, 1)
        
        for i in range(self.klg.N):

            self.y[i, :] = mu[max_idcs[i]]

    def ready(self, use_cuda, gpu_id):

        if use_cuda:

            self.y = self.y.cuda(gpu_id)

        self.y = Variable(self.y, requires_grad=True)

        self.klg.ready(use_cuda, gpu_id)
        self.clf.ready(use_cuda, gpu_id)

    @property
    def params(self):

        return [self.y] + self.clf.params

    def loss(self, a, b):

        y = self.y
        log_p = self.klg.log_p

        loss = sym_kl_div_with_log_input(
            self.clf.log_posterior(y[a:b, :], normalize=True), log_p[a:b, :])

        return loss

    def train(self, num_epoch, batch_size, lrs, verbose_skip, do_annealing):

        N = self.klg.N
        C = self.klg.C
        lr_cond, lr_y, lr_prior = lrs

        optimizer_cond  = optim.Adam([self.clf.like_dist.params[0]], lr=lr_cond)
        optimizer_y     = optim.Adam([self.y],                       lr=lr_y)
        optimizer_prior = optim.Adam(self.clf.prior_dist.params,     lr=lr_prior)

        batches = map(lambda s: (s, s + batch_size) if s + batch_size <= N else (s, N),
                    torch.arange(0, N, batch_size).int())
        batch_num = len(batches)

        log = dict()
        log["loss"] = []

        t = time.time()

        print("------+-------")
        print("Epoch |   Loss")
        print("------+-------")

        for epoch in range(1, num_epoch + 1):

            if do_annealing:

                if epoch < 1000:
                    T = 1 + 9 * (1000 - epoch) / 1000
                elif epoch == 1000:
                    T = 1
                    # Align
                #     y = torch.zeros([self.klg.N, self.clf.like_dist.D])
                #     mu = self.clf.like_dist.params[0].data
                #     _, max_idcs = torch.max(self.klg.p.data, 1)

                #     for i in range(self.klg.N):

                #         y[i, :] = mu[max_idcs[i]]
                #     self.y = Variable(y, requires_grad=True).cuda()
                    
                if epoch <= 1000:
                    logit_div_by_T = self.klg.logit / T
                    p = torch.exp(logit_div_by_T) / \
                        torch.sum(torch.exp(logit_div_by_T), 1).view(N, 1).expand(N, C)

                    self.klg.log_p = logit_div_by_T - \
                        log_sum_exp_stable_mat(logit_div_by_T)
            
            loss_run = 0
            iter_num = 0

            for i, (a, b) in enumerate(batches, 0):

                optimizer_cond.zero_grad()
                optimizer_y.zero_grad()
                optimizer_prior.zero_grad()
        
                loss = self.loss(a, b)
                loss.backward()

                optimizer_y.step()
                optimizer_prior.step()
                optimizer_cond.step() 
                
                loss_run += loss.data[0]
                iter_num += 1
                
            log["loss"].append(loss_run / iter_num)
            
            if epoch % verbose_skip == 0:
                
                print(" %4d | %4.3f" % (epoch, loss_run / iter_num))

        print("Time used: %f" % ( (time.time() - t) ))

        self.log = log

    def plot_loss(self, output_path):

        log = self.log

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        p_loss = ax.plot(log["loss"])
        plt.xlabel("#epoch")
        plt.ylabel("loss")
        plt.legend([p_loss], ["loss"])
        plt.title("Loss v.s. #iterations")

        fig.savefig(output_path + "loss.png")

        return fig, ax

    def plot_mono(self, output_path):

        y = self.y.cpu().data.numpy()
        mu = self.clf.params[0].cpu().data.numpy()

        fig = plt.figure(figsize=(9/1.5, 9/1.5))
        ax = fig.add_subplot(111)
        ax.scatter(y[:,0], y[:,1], alpha=0.7)
        ax.scatter(mu[:,0], mu[:,1], s=64, marker='*')

        fig.savefig(output_path + "mono.png")

        pp = PdfPages(output_path + "mono.pdf")
        fig.savefig(pp, format='pdf')
        pp.close()
        

        return fig, ax

    def plot_color(self, output_path, label_str):

        colors = cm.rainbow(np.linspace(0, 1, self.clf.like_dist.C))
        # colors = plt.get_cmap("tab{0}".format(self.clf.like_dist.C)).colors

        if label_str == "":
            classes = map(lambda i:str(i), range(C))
        else:

            classes = label_str.split(" ")

        y = self.y.cpu().data.numpy()
        mu = self.clf.params[0].cpu().data.numpy()
        # label = self.klg.label.numpy()
        label = np.argmax(self.klg.p.data.cpu().numpy(), axis=1)
        wrong = self.klg.wid_np

        selects = list(range(self.clf.like_dist.C))

        fig = plt.figure(figsize=(9/1.75, 9/1.75))
        ax = fig.add_subplot(111)
        handles = []

        for i in selects:
            idcs = (label == i)
            idcs_wrong = (label == i) & wrong
            handles.append(
                plt.scatter(y[idcs, 0], y[idcs, 1], 
                            marker='.', color=colors[i], alpha=0.5, label="{0}".format(classes[i])))
            handles.append(
                plt.scatter(y[idcs_wrong, 0], y[idcs_wrong, 1], alpha=0.5, s=(32+16)/2,
                            marker='x', color=colors[i]))

            # handles.append(
            #     plt.scatter(mu[i, 0], mu[i, 1], s=256,
            #                 marker='*', color=colors[i], edgecolor='black', label="{0}".format(classes[i])))

        plt.legend(handles=handles, loc='lower left', ncol=2)
        # plt.xlabel("Dim 1")
        # plt.ylabel("Dim 2")
        # plt.title("Low dimensional representation of data")

        fig.savefig(output_path + "color.png")

        pp = PdfPages(output_path + "color.pdf")
        fig.savefig(pp, format='pdf')
        pp.close()
        
        return fig, ax

    def plot_contour(self, output_path, label_str, use_cuda, gpu_id, delta=0.025):

        delta = delta
        y = self.y.cpu().data.numpy()
        ranges = [np.min(y[:, 0]), np.max(y[:, 0]),
                  np.min(y[:, 1]), np.max(y[:, 1])]
        x_ = np.arange(ranges[0], ranges[1], delta)
        y_ = np.arange(ranges[2], ranges[3], delta)
        X, Y = np.meshgrid(x_, y_)

        X_tensor = torch.from_numpy(X)
        a, b = X_tensor.size()
        X_tensor = X_tensor.view(a * b, 1)

        Y_tensor = torch.from_numpy(Y)
        a, b = Y_tensor.size()
        Y_tensor = Y_tensor.view(a * b, 1)

        # Convert to y
        y_grid = torch.cat([X_tensor, Y_tensor], dim=1)

        y_grid = y_grid.float()
        if use_cuda:
            y_grid = y_grid.cuda(gpu_id)
        else:
            y_grid = y_grid.cpu()
        y_grid = Variable(y_grid, requires_grad=False)

        # Compute p_x
        _, p_x = self.clf.posterior(y_grid, return_Z=True)
        p_x = p_x.view(a * b, 1)

        # Transform from 1D to 2D for plotting
        X_tensor = X_tensor.view(a, b)
        Y_tensor = Y_tensor.view(a, b)
        p_x = p_x.view(a, b)

        # Create a simple contour plot with labels using default colors.  The
        # inline argument to clabel will control whether the labels are draw
        # over the line segments of the contour, removing the lines beneath
        # the label
        fig, ax = self.plot_color(output_path, label_str)
        p_contour = plt.contour(X_tensor.cpu().numpy(), Y_tensor.cpu().numpy(), p_x.data.cpu().numpy(), alpha=0.5)
        plt.clabel(p_contour, inline=1, fontsize=7)
        plt.title(ax.title.get_text() + " with contour of P(x)")
        fig.savefig(output_path + "contour.png")

    def evaluate(self, output_path, use_cuda, gpu_id):

        N = self.klg.N
        label = self.klg.label
        p = self.klg.p
        y = self.y
        logit = self.klg.logit
        wrong = self.klg.wid_np

        y_miss_var = Variable(torch.from_numpy(y.data.cpu().numpy()[wrong,:]))
        p_miss_var = Variable(torch.from_numpy(p.data.cpu().numpy()[wrong,:]))
        logit_miss_var = Variable(torch.from_numpy(logit.data.cpu().numpy()[wrong,:]))
        
        if use_cuda:
            y_miss_var = y_miss_var.cuda(gpu_id)
            p_miss_var = p_miss_var.cuda(gpu_id)
            logit_miss_var = logit_miss_var.cuda(gpu_id)

        post = self.clf.posterior(y)
        log_post = self.clf.log_posterior(y)
        post_miss = self.clf.posterior(y_miss_var)
        log_post_miss = self.clf.log_posterior(y_miss_var)

        evaluation = dict()

        # loss_final = self.loss(0, N)
        # print("Final loss: %.3f" % (loss_final.data[0]))
        # evaluation["loss_final"] = loss_final.data[0]

        _, label_pred = torch.max(post, 1)
        label_pred_np = label_pred.data.cpu().numpy()
        acc_ground = float(np.sum(label_pred_np == label.numpy())) / N
        print("Accuracy (ground): %.3f" % (acc_ground))
        evaluation["acc_ground"] = acc_ground
            
        _, label_teacher = torch.max(logit, 1)
        label_teacher_np = label_teacher.data.cpu().numpy()
        acc_teacher = float(np.sum(label_pred_np == label_teacher_np)) / N
        print("Accuracy (teacher): %.3f" % (acc_teacher))
        evaluation["acc_teacher"] = acc_teacher

        evaluation.update(evaluate(post, p, post_miss, p_miss_var,
                                   log_post, logit, log_post_miss, logit_miss_var))

        with open(output_path + 'evaluation.csv', 'wb') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, evaluation.keys())
            w.writeheader()
            w.writerow(evaluation)

        return evaluation

    def output(self, output_path):
        y = self.y
        N = self.klg.N
        label = self.klg.label

        p_y_c, p_x = self.clf.posterior(y, return_Z=True)
        ids = Variable(torch.arange(0, N).view(N,1).float())

        res = torch.cat([ids,
                        y.cpu(),
                        label.view(N,1).float(), 
                        p_x.cpu(),
                        p_y_c.cpu()], 
                        1)
        res = res.data.cpu().numpy()

        np.savetxt(output_path + "result.csv", 
                    res, 
                    delimiter=',', 
                    header="id,dim1,dim2,label,p_x," + \
                           ",".join(map(lambda i: "p_x_" + str(i), range(10))), 
                    comments="")
