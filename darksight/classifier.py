### Load libraries

import torch

from helper import *

### End of library loading

class NaiveBayes:

    def __init__(self, like_dist, prior_dist):

        self.like_dist = like_dist
        self.prior_dist = prior_dist

    def ready(self, use_cuda, gpu_id):

        self.like_dist.ready(use_cuda, gpu_id)
        self.prior_dist.ready(use_cuda, gpu_id)

    @property
    def params(self):

        return self.like_dist.params + self.prior_dist.params

    def posterior(self, y, return_Z=False):

        n, _ = y.size()
        C = self.like_dist.C

        like = self.like_dist.pdf(y)
        prior = self.prior_dist.pdf()

        Z = torch.matmul(like, prior).view(n, 1)

        posterior = like * prior.expand(n, C) / Z.expand(n, C)

        if not return_Z:

            return posterior

        else:

            return posterior, Z

    def log_posterior(self, y, normalize=False):

        n, _ = y.size()
        C = self.like_dist.C

        loglike = self.like_dist.logpdf(y)
        logprior = self.prior_dist.logpdf()

        log_posterior = loglike + logprior

        if normalize:
            
            logZ = torch.log(torch.matmul(torch.exp(loglike), torch.exp(logprior))).view(n,1)
            log_posterior -= logZ

        return log_posterior
