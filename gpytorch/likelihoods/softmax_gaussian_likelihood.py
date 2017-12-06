import math
import gpytorch
import torch
from torch import nn
from torch.autograd import Variable
from gpytorch.random_variables import GaussianRandomVariable, CategoricalRandomVariable
from .likelihood import Likelihood


class SoftmaxGaussianLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """
    def __init__(self, n_features, n_classes, rank=5):
        super(SoftmaxGaussianLikelihood, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.rank = rank

        feature_mixing_weights = nn.Parameter(torch.randn(rank, n_features))
        class_variational_mean = nn.Parameter(torch.randn(n_classes, rank) / rank)
        prior_covar = torch.eye(self.rank, self.rank).unsqueeze(0).expand(self.n_classes, self.rank, self.rank)
        class_chol_variational_covar = nn.Parameter(torch.randn(self.rank, self.rank).triu().expand(self.n_classes, self.rank, self.rank))
        self.register_parameter('feature_mixing_weights', feature_mixing_weights, bounds=(-2, 2))
        self.register_parameter('class_variational_mean', class_variational_mean, bounds=(-2, 2))
        self.register_parameter('class_chol_variational_covar', class_chol_variational_covar, bounds=(-2, 2))

        # Prior
        self.register_buffer('prior_mean', torch.zeros(self.class_variational_mean.size()))
        self.register_buffer('prior_covar', prior_covar)

    def _variational_samples(self, n_samples):
        chol_variational_covar = self.class_chol_variational_covar
        batch_size, diag_size, _ = chol_variational_covar.size()

        # Batch mode
        chol_variational_covar_size = list(chol_variational_covar.size())[-2:]
        mask = chol_variational_covar.data.new(*chol_variational_covar_size).fill_(1).triu()
        mask = Variable(mask.unsqueeze(0).expand(*([chol_variational_covar.size(0)] + chol_variational_covar_size)))

        batch_index = chol_variational_covar.data.new(batch_size).long()
        torch.arange(0, batch_size, out=batch_index)
        batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
        diag_index = chol_variational_covar.data.new(diag_size).long()
        torch.arange(0, diag_size, out=diag_index)
        diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
        diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)

        chol_variational_covar = chol_variational_covar.mul(mask)
        inside = diag.sign().unsqueeze(-1).expand_as(chol_variational_covar).mul(mask)
        chol_variational_covar = inside.mul(chol_variational_covar)

        base_samples = Variable(self.prior_mean.new(n_samples, 1, self.rank, 1).normal_())
        samples = chol_variational_covar.unsqueeze(0).matmul(base_samples).squeeze(-1)
        samples = samples + self.class_variational_mean.unsqueeze(0)
        # n_samples x n_classes x rank
        return samples

    def kl_div(self):
        mean_diffs = Variable(self.prior_mean) - self.class_variational_mean
        chol_variational_covar = self.class_chol_variational_covar
        batch_size, diag_size, _ = chol_variational_covar.size()

        # Batch mode
        chol_variational_covar_size = list(chol_variational_covar.size())[-2:]
        mask = chol_variational_covar.data.new(*chol_variational_covar_size).fill_(1).triu()
        mask = Variable(mask.unsqueeze(0).expand(*([chol_variational_covar.size(0)] + chol_variational_covar_size)))

        batch_index = chol_variational_covar.data.new(batch_size).long()
        torch.arange(0, batch_size, out=batch_index)
        batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
        diag_index = chol_variational_covar.data.new(diag_size).long()
        torch.arange(0, diag_size, out=diag_index)
        diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
        diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)

        chol_variational_covar = chol_variational_covar.mul(mask)
        inside = diag.sign().unsqueeze(-1).expand_as(chol_variational_covar).mul(mask)
        chol_variational_covar = inside.mul(chol_variational_covar)

        batch_size, diag_size, _ = chol_variational_covar.size()
        batch_index = chol_variational_covar.data.new(batch_size).long()
        torch.arange(0, batch_size, out=batch_index)
        batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
        diag_index = chol_variational_covar.data.new(diag_size).long()
        torch.arange(0, diag_size, out=diag_index)
        diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
        matrix_diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)

        logdet_variational_covar = matrix_diag.log().sum() * 2
        trace_logdet_quad_form = gpytorch.trace_logdet_quad_form(mean_diffs, chol_variational_covar,
                                                                 Variable(self.prior_covar))

        # Compute the KL Divergence.
        res = 0.5 * (trace_logdet_quad_form - logdet_variational_covar - len(mean_diffs))
        return res

    def forward(self, input):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:

            p(y|x) = \int p(y|f)p(f|x) df

        Given that p(y=1|f) = \Phi(f), this integral is analytically tractable,
        and if \mu_f and \sigma^2_f are the mean and variance of p(f|x), the
        solution is given by:

            p(y|x) = \Phi(\frac{\mu}{\sqrt{1+\sigma^2_f}})
        """
        if not isinstance(input, GaussianRandomVariable):
            raise RuntimeError(' '.join([
                'SoftmaxGaussianLikelihood expects a Gaussian',
                'distributed latent function to make predictions',
            ]))

        f = input._variational_strategy.variational_samples(input)
        if f.ndimension() != 3:
            raise RuntimeError('f should have 3 dimensions: features x data x samples')
        n_features, n_data, n_samples = f.size()
        if n_features != self.n_features:
            raise RuntimeError('There should be %d features' % self.n_features)

        mixed_fs = self.feature_mixing_weights.matmul(f.view(n_features, n_samples * n_data))
        class_weight_samples = self._variational_samples(n_samples)
        class_fs = class_weight_samples.matmul(mixed_fs.unsqueeze(0)).view(n_samples, self.n_classes, n_data, n_samples)
        class_fs = class_fs.permute(0, 3, 2, 1)
        softmax = nn.functional.softmax(class_fs, 3).mean(1).mean(0)
        return CategoricalRandomVariable(softmax)

    def log_probability(self, f, y):
        """
        Computes the log probability \sum_{i} \log \Phi(y_{i}f_{i}), where
        \Phi(y_{i}f_{i}) is computed by averaging over a set of s samples of
        f_{i} drawn from p(f|x).
        """
        if f.ndimension() != 3:
            raise RuntimeError('f should have 3 dimensions: features x data x samples')
        n_features, n_data, n_samples = f.size()
        if n_features != self.n_features:
            raise RuntimeError('There should be %d features' % self.n_features)

        mixed_fs = self.feature_mixing_weights.matmul(f.view(n_features, n_samples * n_data))
        class_weight_samples = self._variational_samples(n_samples)
        class_fs = class_weight_samples.matmul(mixed_fs.unsqueeze(0)).view(n_samples, self.n_classes, n_data, n_samples)
        class_fs = class_fs.permute(0, 3, 2, 1)
        softmax = nn.functional.softmax(class_fs, 3).mean(1).mean(0)
        log_prob = -nn.functional.nll_loss(softmax.log(), y, size_average=False)
        return log_prob
