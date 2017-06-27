import math
import torch
import numpy as np
from torch.autograd import Variable
from gpytorch.math.functions import ExactGPMarginalLogLikelihood
from gpytorch.lazy import Kron


covar = torch.Tensor([
    [5, -3, 0],
    [-3, 5, 0],
    [0, 0, 2],
])
other_covar = torch.Tensor([
    [3, 0, -3],
    [0, 2, 1],
    [-3, 1, 6],
])
y = torch.randn(3)
kron_y = torch.randn(9)


def test_forward():
    actual = y.dot(covar.inverse().mv(y))
    actual += math.log(np.linalg.det(covar.numpy()))
    actual += math.log(2 * math.pi) * len(y)
    actual *= -0.5

    covarvar = Variable(covar)
    yvar = Variable(y)
    res = ExactGPMarginalLogLikelihood()(covarvar, yvar)
    assert(torch.norm(actual - res.data) < 1e-4)


def test_forward_kron():
    def evaluate_stub(self):
        raise Exception('Evaluate should not be called')

    covarvar = Kron()(Variable(covar), Variable(other_covar), Variable(torch.zeros(1)))
    yvar = Variable(kron_y)
    actual = ExactGPMarginalLogLikelihood()(covarvar.evaluate(), yvar)

    covarvar = Kron()(Variable(covar), Variable(other_covar), Variable(torch.zeros(1)))
    covarvar.evaluate = evaluate_stub.__get__(covarvar, covarvar.__class__)
    yvar = Variable(kron_y)
    res = ExactGPMarginalLogLikelihood()(covarvar, yvar)

    assert(torch.norm(actual.data - res.data) < 1e-4)


def test_backward():
    covarvar = Variable(covar, requires_grad=True)
    yvar = Variable(y, requires_grad=True)
    actual_mat_grad = torch.ger(covar.inverse().mv(y), covar.inverse().mv(y))
    actual_mat_grad -= covar.inverse()
    actual_mat_grad *= 0.5
    actual_mat_grad *= 3 # For grad output

    actual_y_grad = -covar.inverse().mv(y)
    actual_y_grad *= 3 # For grad output

    covarvar = Variable(covar, requires_grad=True)
    yvar = Variable(y, requires_grad=True)
    output = ExactGPMarginalLogLikelihood()(covarvar, yvar) * 3
    output.backward()

    assert(torch.norm(actual_mat_grad - covarvar.grad.data) < 1e-4)
    assert(torch.norm(actual_y_grad - yvar.grad.data) < 1e-4)


def test_backward_kron():
    covarvar = Variable(covar, requires_grad=True)
    other_covarvar = Variable(other_covar, requires_grad=True)
    diagvar = Variable(torch.zeros(1), requires_grad=True)
    kronvar = Kron()(covarvar, other_covarvar, diagvar)
    yvar = Variable(kron_y, requires_grad=True)
    actual = ExactGPMarginalLogLikelihood()(kronvar.evaluate(), yvar)
    actual.backward()

    actual_covar_grad = covarvar.grad.data
    actual_other_covar_grad = other_covarvar.grad.data
    actual_diag_grad = diagvar.grad.data
    actual_y_grad = yvar.grad.data

    covarvar = Variable(covar, requires_grad=True)
    other_covarvar = Variable(other_covar, requires_grad=True)
    diagvar = Variable(torch.zeros(1), requires_grad=True)
    kronvar = Kron()(covarvar, other_covarvar, diagvar)
    yvar = Variable(kron_y, requires_grad=True)
    res = ExactGPMarginalLogLikelihood()(kronvar, yvar)
    res.backward()

    res_covar_grad = covarvar.grad.data
    res_other_covar_grad = other_covarvar.grad.data
    res_diag_grad = diagvar.grad.data
    res_y_grad = yvar.grad.data

    assert(torch.norm(actual_covar_grad - res_covar_grad) < 1e-4)
    assert(torch.norm(actual_other_covar_grad - res_other_covar_grad) < 1e-4)
    assert(torch.norm(actual_diag_grad - res_diag_grad) < 1e-4)
    assert(torch.norm(actual_y_grad - res_y_grad) < 1e-4)
