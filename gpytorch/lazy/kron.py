import torch
from torch.autograd import Variable
from gpytorch.lazy import LazyFunction, LazyVariable
from gpytorch.utils.kron import kron_forward, kron_backward


class KronVariable(LazyVariable):
    def check_inputs(self, inputs):
        if len(inputs) != 3:
            raise RuntimeError('Kron variables require 3 inputs')
        if not inputs[2].numel() == 1:
            raise RuntimeError('The last variable must be a single term')


class Kron(LazyFunction):
    variable_class = KronVariable


    def forward(self, mat_1, mat_2, diag):
        if diag.numel() == 1:
            self.diag_val = diag.squeeze()[0]
        else:
            raise RuntimeError('Diag must be a single-element tensor')

        self.save_for_backward(mat_1, mat_2)

        return kron_forward(mat_1, mat_2, self.diag_val)


    def backward(self, grad_output):
        grad_mat_1 = None
        grad_mat_2 = None
        diag_grad = None
        mat_1, mat_2 = self.saved_tensors

        return kron_backward(mat_1, mat_2, grad_output, self.needs_input_grad)
