# DarkSight - Interpreting Deep Classifier by Visual Distillation of Dark Knowledge

DarkSight is a dimension reduction technique to visualize any black-box classifier. 

Please visit [here](http://xuk.ai/darksight) for more information (the related paper, main results, exemplar visualization and demos).

## PyTorch Implementation

This repository contains a PyTorch implementation of DarkSight.

### How to Install

Download or clone this repository and put the folder `darksight` in the same path as the Python file need to import it.

#### Dependencies

- PyTorch
- NumPy
- Matplotlib

### Demo

A [demo](https://github.com/xukai92/darksight/blob/master/demo.ipynb) is provided in this repository to illustrate the basic use of DarkSight, using the provided [output](https://github.com/xukai92/darksight/blob/master/examples/data/lenet-mnist.csv) from a 98.2% accuracy LeNet trained on MNIST dataset.

### Basic APIs

#### `darksight.Knowledge`

`darksight.Knowledge(output, T=1)` is the class that wraps a clssifier's output.

Parameters
- `output`: the classifier's output; should be a `numpy.array`
  - It can be either the predictive probabilities or the logit before softmax

Optional
- `T`: the temperature used when normalizing the provided output

#### `darksight.DarkSight`

`darksight.DarkSight(klg)` is the class for the DarkSight proposed in the paper.
It defines a Naive Bayes model on 2D and performs non-parametric dimension reduction and model compression jointly based on a symmetric KL divergence objective.

Parameters
- `klg` is the knowledge object wrapped by `darksight.Knowledge`

##### `darksight.DarkSight.train(num_epoch, lrs, batch_size=1000, verbose_skip=100, do_annealing=False, annealing_length=1000, highest_T=10, annealing_stepsize=100)`

Parameters
  - `num_epoch`: number of epochs for training
  - `lrs`: learning rates for each component, conditional distribution, low-dimensional embedding and prior distribution, in a list, i.e. `[lr_cond, lr_y, lr_prior]`

Optional
  - `batch_size`: batch size for traning
  - `verbose_skip`: number of epochs for printing training logs
  - `do_annealing`: whether to do annealing or not
  - `annealing_length`: length of epoch for annealing
  - `highest_T`: start temperature of annealing
  - `annearling_stepsize`: the step size of calculating new temperature

##### `darksight.DarkSight.plot_loss()`

Helper function to plot the loss trace.

##### `darksight.DarkSight.plot_y()`

Helper function to print the learnt low-dimensional embeddings as a scatter plot.

Optional
  - `color_on`: color or monotonic?
  - `mu_on`: plot means of each cluster?
  - `labels`: a Python list of string for labels used in the plot
  - `contour_on`: plot contour based on P(y)?
    - Note that generating contour requires running the model so we need to specify whether and how to use GPU for this purpose below
  - `use_cuda`: use GPU?
  - `gpu_id`: which GPU to use?
  - `contour_slices`: how fine do you want your contour to be computed?
  - `contour_num`: how many contour levels you want?

##### `darksight.DarkSight.y`

Low-dimensional embedding learnt.

##### `darksight.DarkSight.mu`: 

Means of each conditional distribution learnt.

##### `darksight.DarkSight.H`: 

Precision matrix of each conditional distribution learnt.

##### `darksight.DarkSight.w`: 

Parameters of the prior distribution.

##### `darksight.DarkSight.output(output_file_path)`

Helper function to output the training results.

Parameters
  - `output_file_path`: output file path

## Contact

Kai Xu is the first author of the corresponding paper and the maintainer of this library, feel free to contact him on the paper or the library by email: xu.kai@ed.ac.uk.