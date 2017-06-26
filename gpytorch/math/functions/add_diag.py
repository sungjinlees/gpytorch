import torch
from torch.autograd import Function, Variable
from gpytorch.lazy import Kron, KronVariable, register_lazy_function


class AddDiag(Function):
    def forward(self, input, diag):
        if diag.numel() != 1:
            raise RuntimeError('Input must be a single-element tensor')
        val = diag.squeeze()[0]

        return torch.eye(*input.size()).mul_(val).add_(input)


    def backward(self, grad_output):
        input_grad = None
        diag_grad = None

        if self.needs_input_grad[0]:
            input_grad = grad_output

        if self.needs_input_grad[1]:
            diag_grad = grad_output.new().resize_(1)
            diag_grad.fill_(grad_output.trace())

        return input_grad, diag_grad


class KronAddDiag(Kron):
    def __call__(self, mat_a, mat_b, diag, new_diag_term, **kwargs):
        return Kron()(mat_a, mat_b, diag + new_diag_term)


register_lazy_function(AddDiag, (KronVariable, Variable), KronAddDiag)
