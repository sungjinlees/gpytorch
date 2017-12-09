import gpytorch
import torch
from torch import nn
from torch.autograd import Variable
from .grid_inducing_point_module import InducingPointModule
from ..lazy import LazyVariable, NonLazyVariable, MatmulLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import InducingPointStrategy
from ..utils import left_interp


class AdditiveInducingPointModule(InducingPointModule):
    def __init__(self, inducing_points, n_components, sum_output=False):
        super(AdditiveInducingPointModule, self).__init__(inducing_points)
        self.n_components = n_components
        self.sum_output = sum_output

        # Resize variational parameters to have one size per component
        variational_mean = self.variational_mean
        chol_variational_covar = self.chol_variational_covar
        variational_mean.data.resize_(*([n_components] + list(variational_mean.size())))
        chol_variational_covar.data.resize_(*([n_components] + list(chol_variational_covar.size())))

    def __call__(self, inputs, **kwargs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(-1).unsqueeze(-1)
        elif inputs.ndimension() == 2:
            inputs = inputs.unsqueeze(-1)
        elif inputs.ndimension() != 3:
            raise RuntimeError('AdditiveInducingPointModule expects a 3d tensor.')

        n_data, n_components, n_dimensions = inputs.size()
        if n_components != self.n_components:
            raise RuntimeError('The number of components should match the number specified.')
        inputs = inputs.transpose(0, 1).contiguous()

        inducing_points = self._inducing_points
        n_data, n_components, n_dimensions = inducing_points.size()
        if n_components != self.n_components:
            raise RuntimeError('The number of components should match the number specified.')
        inducing_points = inducing_points.transpose(0, 1).contiguous()


        if self.exact_inference:
            raise RuntimeError

        variational_mean = self.variational_mean
        chol_variational_covar = self.chol_variational_covar

        # Initialize variational parameters, if necessary
        if self.training:
            if not torch.equal(inputs.data, inducing_points):
                raise RuntimeError('At the moment, we assume that the inducing_points are the'
                                   ' training inputs.')
            inducing_output = gpytorch.Module.__call__(self, Variable(inducing_points))
            output = inducing_output

            if not self.variational_params_initialized[0]:
                mean_init = output.mean().data
                mean_init_size = list(mean_init.size())
                mean_init_size[0] = self.n_components
                mean_init = mean_init.expand(*mean_init_size)
                chol_covar_init = torch.eye(mean_init.size(-1)).type_as(mean_init).unsqueeze(0)
                chol_covar_init_size = list(chol_covar_init.size())
                chol_covar_init_size[0] = self.n_components
                chol_covar_init = chol_covar_init.expand(*chol_covar_init_size)

                variational_mean.data.copy_(mean_init)
                chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)

        else:
            n_induc = inducing_points.size(1)
            full_inputs = torch.cat([Variable(inducing_points), inputs], 1)
            full_output = super(InducingPointModule, self).__call__(full_inputs)
            full_mean, full_covar = full_output.representation()

            induc_mean = full_mean[:, :n_induc]
            test_mean = full_mean[:, n_induc:]
            induc_induc_covar = full_covar[:, :n_induc, :n_induc]
            induc_test_covar = full_covar[:, :n_induc, n_induc:]
            test_induc_covar = full_covar[:, n_induc:, :n_induc]
            test_test_covar = full_covar[:, n_induc:, n_induc:]

            alpha = gpytorch.inv_matmul(induc_induc_covar, variational_mean - induc_mean).unsqueeze(-1)
            test_mean = torch.add(test_mean, test_induc_covar.matmul(alpha).squeeze(-1))
            if self.sum_output:
                test_mean = test_mean.sum(0)

            # Compute test covar
            if isinstance(induc_test_covar, LazyVariable):
                induc_test_covar = induc_test_covar.evaluate()
            inv_product = gpytorch.inv_matmul(induc_induc_covar, induc_test_covar)
            factor = chol_variational_covar.matmul(inv_product)
            test_covar = MatmulLazyVariable(factor.transpose(-1, -2), factor)

            inducing_output = GaussianRandomVariable(induc_mean, induc_induc_covar)
            output = GaussianRandomVariable(test_mean, test_covar)

        # Add variational strategy
        output._variational_strategy = InducingPointStrategy(variational_mean,
                                                             chol_variational_covar,
                                                             inducing_output)

        if not isinstance(output, GaussianRandomVariable):
            raise RuntimeError('Output should be a GaussianRandomVariable')

        return output
