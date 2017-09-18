import torch
import math
import numpy as np
import gpytorch
from gpytorch import utils
from gpytorch.utils.mubs_trace import mubs_trace, hutchinson_trace
from gpytorch.utils.toeplitz import interpolated_sym_toeplitz_matmul
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
import pdb

N = 500

x = Variable(torch.linspace(0, 5, N))


class Model(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
        super(Model, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        covar_module = RBFKernel(log_lengthscale_bounds=(-3, -3))
        self.grid_covar_module = GridInterpolationKernel(covar_module)
        self.initialize_interpolation_grid(1000, grid_bounds=[(0, 5)])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.grid_covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


prior_observation_model = Model()
pred = prior_observation_model(x)
lazy_toeplitz_var_no_diag = prior_observation_model.grid_covar_module(x)
lazy_toeplitz_var = pred.covar()
T = utils.toeplitz.sym_toeplitz(lazy_toeplitz_var.c.data)
W_left = utils.toeplitz.index_coef_to_sparse(lazy_toeplitz_var.J_left,
                                             lazy_toeplitz_var.C_left,
                                             len(lazy_toeplitz_var.c))
W_right = utils.toeplitz.index_coef_to_sparse(lazy_toeplitz_var.J_right,
                                              lazy_toeplitz_var.C_right,
                                              len(lazy_toeplitz_var.c))

WTW = torch.dsmm(W_right, torch.dsmm(W_left, T).t()) + torch.diag(lazy_toeplitz_var.added_diag.data)
print WTW
clos = lambda v: interpolated_sym_toeplitz_matmul(lazy_toeplitz_var.c.data, v, W_left, W_right, lazy_toeplitz_var.added_diag.data)

t1 = [math.fabs(hutchinson_trace(clos, N, 50) - WTW.trace()) / WTW.trace() for i in range(500)]
t2 = [math.fabs(mubs_trace(clos, N, 50) - WTW.trace()) / WTW.trace() for i in range(500)]

print 'Interpolated Toeplitz Trace:', np.mean(t1), np.mean(t2)

A = torch.randn(N, N)
A = A.t().matmul(A)
A = A / torch.norm(A)

clos2 = lambda v: A.matmul(v)

t3 = [math.fabs(hutchinson_trace(clos2, N, 50) - A.trace()) / A.trace() for i in range(500)]
t4 = [math.fabs(mubs_trace(clos2, N, 50) - A.trace()) / A.trace() for i in range(500)]

print 'Standard Matrix Trace:', np.mean(t3), np.mean(t4)