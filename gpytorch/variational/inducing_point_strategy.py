import torch
import gpytorch
from torch.autograd import Variable
from ..lazy import MatmulLazyVariable
from .variational_strategy import VariationalStrategy
from ..utils import StochasticLQ


class InducingPointStrategy(VariationalStrategy):
    def variational_samples(self, output, n_samples=None):
        if n_samples is None:
            n_samples = gpytorch.functions.num_trace_samples

        # Draw samplse from variational distribution
        base_samples = Variable(self.variational_mean.data.new(self.variational_mean.size(-1), n_samples).normal_())
        if self.variational_mean.ndimension() > 1:
            # Batch mode
            base_samples = base_samples.unsqueeze(0)

        if output == self.inducing_output:
            samples = self.chol_variational_covar.transpose(-1, -2).matmul(base_samples)
            samples = samples + self.variational_mean.unsqueeze(-1)
            return samples
        elif isinstance(output.covar(), MatmulLazyVariable):
            covar = output.covar()
            chol_covar = covar.lhs

            samples = chol_covar.matmul(base_samples)
            samples = samples + output.mean().unsqueeze(-1)
            return samples
        else:
            raise RuntimeError
