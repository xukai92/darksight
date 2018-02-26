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
from distributions import *
from classifier import *
from darksight import *


### Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('logit_file', type=str, metavar='LOGIT_PATH',
                    help='path of logit.pt')
parser.add_argument('data_file', type=str, metavar='DATA_PATH',
                    help='path of data.pt')
parser.add_argument('output_dir', type=str, metavar='OUTPUT',
                    help='path to output')
parser.add_argument('--labels', type=str, default="", metavar='L',
                    help='input batch size for training (default: "")')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--temperature', type=int, default=1, metavar='T',
                    help='temperature of logit to probability (default: 1)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--epochs-p2', type=int, default=1000, metavar='N',
                    help='number of epochs in phase 2 to train (default: 1000)')
parser.add_argument('--dist', type=str, default="t", metavar='DIST',
                    help='type of conditional distribution to use (default: t)')
parser.add_argument('--lr-cond', type=float, default=5e-3, metavar='LR',
                    help='learning rate for cond (default: 5e-3)')
parser.add_argument('--lr-y', type=float, default=5e-0, metavar='LR',
                    help='learning rate for y (default: 5e-0)')
parser.add_argument('--lr-prior', type=float, default=1e-5, metavar='LR',
                    help='learning rate for prior (default: 1e-5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-train', action='store_true', default=False,
                    help='do training or not; if not, read model from output')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='store model to .pt file')
parser.add_argument('--save-csv', action='store_true', default=False,
                    help='store results to .csv file')
parser.add_argument('--gpu-id', type=int, default=0, metavar='G',
                    help='gpu id (default: 0)')
parser.add_argument('--seed', type=int, default=777, metavar='S',
                    help='random seed (default: 777)')
parser.add_argument('--verbose-skip', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--do-overwrite', action='store_true', default=False,
                    help='overwrite by default otherwise ask user')
parser.add_argument('--do-annealing', action='store_true', default=False,
                    help='do annealing during training')

args = parser.parse_args()
print("Running with args: ")
for arg in vars(args):
    print "  ", arg, ":=", getattr(args, arg)

output_path  = args.output_dir
use_cuda     = not args.no_cuda
gpu_id       = args.gpu_id
batch_size   = args.batch_size
T            = args.temperature
dist         = args.dist
verbose_skip = args.verbose_skip
save_model   = args.save_model
lr_cond      = args.lr_cond
lr_y         = args.lr_y
lr_prior     = args.lr_prior
logit_pt_str = args.logit_file
data_pt_str  = args.data_file
epochs       = args.epochs
epochs_p2    = args.epochs_p2
label_str    = args.labels
do_annealing = args.do_annealing

# Deal with output folder
if not output_path[-1] == '/':
    output_path += '/'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
else:
    if not args.do_overwrite:
        print("Output already exists, files may be overwritten, still go ahead? (y/n)")
        ans = sys.stdin.readline()
        if ans != "y\n":
            print("Bye~")
            exit()

# Abort if no GPU when use_gpu == True
if use_cuda:
    assert torch.cuda.is_available(), "No GPU available"

# Set random seeds
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)




klg = Knowledge(logit_pt_str, data_pt_str)

D = 2

if dist == "t":
    like = Student(klg.C, D)
elif dist == "g":
    like = Gaussian(klg.C, D)
else:
    print("Unknow dist: {0}".format(dist))
    exit()

prior = Softmax(klg.C)
clf = CondX(like, prior)

ds = DarkSight(klg, clf)

if not args.no_train:
    
    # ds.align()
    ds.ready(use_cuda, gpu_id)

    ds.train([epochs, epochs_p2], batch_size, 
             [lr_cond, lr_y, lr_prior], verbose_skip, do_annealing)
    ds.plot_loss(output_path)

else:

    y, mu, H, W = torch.load(output_path + "model.pt")

    like.params[0] = mu
    like.params[1] = H
    prior.params[0] = W

    ds.y = y

    ds.ready(use_cuda, gpu_id)

if (not args.no_train) and save_model:
    torch.save(map(lambda p: p.data.cpu(), ds.params), output_path + "model.pt")


ds.plot_mono(output_path)
ds.plot_color(output_path, label_str)
ds.plot_contour(output_path, label_str, use_cuda, gpu_id)
evaluation = ds.evaluate(output_path, use_cuda, gpu_id)

if args.save_csv:
    ds.output(output_path)

if not args.no_train:
    with open(output_path + "args.txt", 'w') as the_file:
        the_file.write(str(args))

log = vars(args)
log.update(evaluation)

log_experiment(log, "DarkSight")
