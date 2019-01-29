### Load libraries

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')         # for better looking
import matplotlib.cm as cm      # for generating color list
matplotlib.rcParams['xtick.direction'] = 'out'  # let x ticks to behind x-axis
matplotlib.rcParams['ytick.direction'] = 'out'  # let y ticks to left y-axis
from matplotlib.backends.backend_pdf import PdfPages

# Utility
import time

# Sci computing
import numpy as np
import torch
import torch.optim as optim

from helper import *
from distributions import *
from classifier import *

### End of library loading



class Knowledge:

    def __init__(self, logit_np, T=1):

        self.T = T

        logit = torch.from_numpy(logit_np).float()

        print("[Knowledge.__init__] {0} with size of {1} is loaded".format(type(logit), logit.size()))

        # Generate teacher's label
        self.label_pred_np = np.argmax(logit.numpy(), axis=1)

        # C - #classes
        # N - #data points
        N, C = logit.size()

        # Convert logit to probability
        logit_div_by_T = logit / T
        p = torch.exp(logit_div_by_T) / torch.sum(torch.exp(logit_div_by_T), 1).view(N, 1).expand(N,C)

        # Log for numerical stability
        log_p = logit_div_by_T - torch.logsumexp(logit_div_by_T, 1, keepdim=True)

        self.N = N
        self.C = C

        self.logit = logit
        self.log_p = log_p

    def ready(self, use_cuda, gpu_id):

        if use_cuda:

            self.logit = self.logit.cuda(gpu_id)
            self.log_p = self.log_p.cuda(gpu_id)



class DarkSightGeneric:

    def __init__(self, klg, clf):
        '''
        @input
            klg         :   knowledge
            clf         :   low-dimensional classifier
        '''

        self._y = torch.zeros([klg.N, clf.like_dist.D])
        self.klg = klg
        self.clf = clf

    def align(self):

        mu = self.clf.like_dist.params[0]
        _, max_idcs = torch.max(self.klg.log_p, 1)

        for i in range(self.klg.N):

            self._y[i, :] = mu[max_idcs[i]]

    def ready(self, use_cuda=True, gpu_id=0):

        if use_cuda:

            self._y = self._y.cuda(gpu_id)

        self._y.requires_grad_()

        self.klg.ready(use_cuda, gpu_id)
        self.clf.ready(use_cuda, gpu_id)

    @property
    def params(self):

        return [self._y] + self.clf.params

    @property
    def y(self):

        return self._y.cpu().data.numpy()

    def loss(self, a, b):

        y = self._y
        log_p = self.klg.log_p

        loss = sym_kl_div_with_log_input(
            self.clf.log_posterior(y[a:b, :], normalize=True), log_p[a:b, :])

        return loss

    def train(self, num_epoch, lrs, batch_size=1000, verbose_skip=100,
                    do_annealing=False, annealing_length=1000, highest_T=10, annealing_stepsize=100):

        N = self.klg.N
        C = self.klg.C
        lr_cond, lr_y, lr_prior = lrs

        optimizer_cond  = optim.Adam([self.clf.like_dist.params[0]], lr=lr_cond)
        optimizer_y     = optim.Adam([self._y],                       lr=lr_y)
        optimizer_prior = optim.Adam(self.clf.prior_dist.params,     lr=lr_prior)

        batches = map(lambda s: (s, s + batch_size) if s + batch_size <= N else (s, N),
                    torch.arange(0, N, batch_size).int())
        batch_num = len(batches)

        log = dict()
        log["loss"] = []

        t = time.time()

        print("-------+--------+---")
        print(" Epoch |  Loss  | T ")
        print("-------+--------+---")

        for epoch in range(1, num_epoch + 1):

            if do_annealing:

                target_T = self.klg.T

                if epoch < annealing_length:

                    count_down_epoch = annealing_length - epoch + 1
                    T = target_T + (highest_T - target_T) * count_down_epoch / annealing_length
                    do_update = (count_down_epoch % annealing_stepsize) == 0

                elif epoch == annealing_length:

                    T = target_T
                    do_update = True

                else:

                    do_update = False

                if do_update:

                    logit_div_by_T = self.klg.logit / T

                    p = torch.exp(logit_div_by_T) / \
                        torch.sum(torch.exp(logit_div_by_T), 1).view(N, 1).expand(N, C)

                    self.klg.log_p = logit_div_by_T - torch.logsumexp(logit_div_by_T, 1, keepdim=True)
            else:

                T = self.klg.T

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

                loss_run += loss.item()
                iter_num += 1

            log["loss"].append(loss_run / iter_num)

            if epoch % verbose_skip == 0:

                print(" %5d | %5.4f | %d" % (epoch, loss_run / iter_num, T))

        print("Time used: %f" % (time.time() - t))

        print("Final loss: %f" % (log["loss"][-1]))
        print("Accuracy to teacher: %5.4f" % (self.acc_teacher() * 100))

        self.log = log

    def acc_teacher(self):

        N = self.klg.N
        label_pred_np = self.klg.label_pred_np
        y = self._y

        post = self.clf.posterior(y)

        _, label_pred = torch.max(post, 1)
        label_pred_np = label_pred.data.cpu().numpy()

        acc = float(np.sum(label_pred_np == label_pred_np)) / N

        return acc

    def plot_loss(self):

        log = self.log

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)

        p_loss = plt.plot(log["loss"])
        plt.xlabel("#epoch")
        plt.ylabel("Loss")
        plt.legend([p_loss], ["loss"])
        plt.title("Loss v.s. #iterations")

        return fig, ax

    def plot_y(self, color_on=True, mu_on=True, contour_on=False, labels=None,
                     use_cuda=True, gpu_id=0, contour_slices=100, contour_num=5):

        y_np = self._y.data.cpu().numpy()
        mu_np = self.clf.like_dist.params[0].cpu().data.numpy()
        label_pred_np = self.klg.label_pred_np
        C = self.klg.C

        if not labels:

            labels = list(range(C))

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)

        if color_on:

            colors = color_list = plt.cm.tab10(np.linspace(0, 1, C))

            for i in range(C):
                mask_i = (label_pred_np == i)
                plt.scatter(y_np[mask_i, 0], y_np[mask_i, 1], c=colors[i], label=labels[i], alpha=.9)

        else:

            plt.scatter(y_np[:,0], y_np[:,1], alpha=.9, label="y")

        if mu_on:

            plt.scatter(mu_np[:,0], mu_np[:,1], marker="*", c="white", alpha=.75,
                        linewidths=1, s=200, edgecolors="black", label=r"$\mu$")

        if contour_on:

            x_ls = np.linspace(np.min(y_np[:, 0]), np.max(y_np[:, 0]), contour_slices)
            y_ls = np.linspace(np.min(y_np[:, 1]), np.max(y_np[:, 1]), contour_slices)
            X, Y = np.meshgrid(x_ls, y_ls)

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
            y_grid.requires_grad_()

            # Compute p_y
            _, p_y = self.clf.posterior(y_grid, return_Z=True)
            p_y = p_y.view(a * b, 1)

            # Transform from 1D to 2D for plotting
            X_tensor = X_tensor.view(a, b)
            X_np = X_tensor.cpu().numpy()
            Y_tensor = Y_tensor.view(a, b)
            Y_np = Y_tensor.cpu().numpy()
            p_y = p_y.view(a, b)
            p_y_np = p_y.data.cpu().numpy()

            p_contour = plt.contour(X_np, Y_np, p_y_np, levels=np.linspace(np.min(p_y_np), np.max(p_y_np), contour_num), alpha=0.5)

        plt.legend(ncol=2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        return fig, ax

    def output(self, output_file_path):
        y = self._y
        N = self.klg.N
        C = self.klg.C
        label = torch.from_numpy(self.klg.label_pred_np)

        p_y_c, p_y = self.clf.posterior(y, return_Z=True)
        ids = torch.arange(0, N).view(N, 1).float()

        res = torch.cat((ids.data,
                        y.data.cpu(),
                        label.view(N, 1).float(),
                        p_y.data.cpu(),
                        p_y_c.data.cpu()),
                        1)
        res = res.cpu().numpy()


        np.savetxt(output_file_path, res,
                   delimiter=',',
                   header="id,dim1,dim2,label_pred,p_y," + \
                           ",".join(map(lambda i: "p_y_" + str(i), range(C))),
                   comments="")

class DarkSight(DarkSightGeneric):

    def __init__(self, klg, D=2):

        cond = Student(klg.C, D)
#         cond = Gaussian(klg.C, D)
        prior = Softmax(klg.C)
        nb = NaiveBayes(cond, prior)

        DarkSightGeneric.__init__(self, klg, nb)

    @property
    def mu(self):

        return self.clf.like_dist.mu.cpu().data.numpy()

    @property
    def H(self):

        return self.clf.like_dist.H.cpu().data.numpy()

    @property
    def w(self):

        return self.clf.prior_dist.w.cpu().data.numpy()
