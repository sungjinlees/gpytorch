import math
import torch
from torch import nn
from gpytorch.random_variables import GaussianRandomVariable, CategoricalRandomVariable
from .likelihood import Likelihood


class SoftmaxCategoricalLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """
    def __init__(self, n_features, n_classes, rank=5):
        super(SoftmaxCategoricalLikelihood, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.rank = rank

        feature_mixing_weights = nn.Parameter(torch.randn(rank, n_features))
        categorical_params_base = nn.Parameter(torch.randn(n_classes, rank))
        self.register_parameter('feature_mixing_weights', feature_mixing_weights, bounds=(-2, 2))
        self.register_parameter('categorical_params_base', categorical_params_base, bounds=(-2, 2))

    @property
    def categorical_params(self):
        return nn.functional.softmax(self.categorical_params_base)

    def categorical_kl_div(self):
        categorical_params = self.categorical_params 
        return torch.sum(categorical_params * categorical_params.log() / -math.log(self.n_classes))

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
                'SoftmaxCategoricalLikelihood expects a Gaussian',
                'distributed latent function to make predictions',
            ]))

        f = input._variational_strategy.variational_samples(input).transpose(1, 2).contiguous()
        if f.ndimension() != 3:
            raise RuntimeError('f should have 3 dimensions: features x data x samples')
        n_features, n_samples, n_data = f.size()
        if n_features != self.n_features:
            raise RuntimeError('There should be %d features' % self.n_features)

        mixed_fs = self.feature_mixing_weights.matmul(f.view(n_features, n_samples * n_data))
        categorical_params = nn.functional.softmax(self.categorical_params_base)
        categorical_samples = CategoricalRandomVariable(categorical_params).sample(n_samples).float()
        cat_fs = categorical_samples.matmul(mixed_fs.unsqueeze(0)).transpose(0, 1).contiguous()
        cat_fs = cat_fs.view(self.n_classes, n_samples * n_samples * n_data)
        softmax = nn.functional.softmax(cat_fs.t()).view(n_samples * n_samples, n_data, self.n_classes)
        softmax = softmax.mean(0)
        return CategoricalRandomVariable(softmax)

    def log_probability(self, f, y):
        """
        Computes the log probability \sum_{i} \log \Phi(y_{i}f_{i}), where
        \Phi(y_{i}f_{i}) is computed by averaging over a set of s samples of
        f_{i} drawn from p(f|x).
        """
        if f.ndimension() != 3:
            raise RuntimeError('f should have 3 dimensions: features x data x samples')
        f = f.transpose(1, 2).contiguous()
        n_features, n_samples, n_data = f.size()
        if n_features != self.n_features:
            raise RuntimeError('There should be %d features' % self.n_features)

        mixed_fs = self.feature_mixing_weights.matmul(f.view(n_features, n_samples * n_data))
        categorical_params = nn.functional.softmax(self.categorical_params_base)
        categorical_samples = CategoricalRandomVariable(categorical_params).sample(n_samples).float()
        cat_fs = categorical_samples.matmul(mixed_fs.unsqueeze(0)).transpose(0, 1).contiguous()
        cat_fs = cat_fs.view(self.n_classes, n_samples * n_samples * n_data)
        
        log_prob = -nn.functional.cross_entropy(cat_fs.t(), y.unsqueeze(1).repeat(n_samples ** 2, 1).view(-1),
                                                size_average=False)
        return log_prob.div(n_samples * n_samples)
