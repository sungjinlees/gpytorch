import torch
from .lazy_variable import LazyVariable


class MatmulLazyVariable(LazyVariable):
    def __init__(self, lhs, rhs):
        super(MatmulLazyVariable, self).__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs

    def _matmul_closure_factory(self, lhs, rhs):
        def closure(tensor):
            return torch.matmul(lhs, rhs).matmul(tensor)

        return closure

    def _derivative_quadratic_form_factory(self, lhs, rhs):
        def closure(left_factor, right_factor):
            left_grad = left_factor.transpose(-1, -2).matmul(right_factor.matmul(rhs.transpose(-1, -2)))
            right_grad = lhs.transpose(-1, -2).matmul(left_factor.transpose(-1, -2)).matmul(right_factor)
            return left_grad, right_grad

        return closure

    def _size(self):
        if self.lhs.ndimension() > 2:
            return torch.Size((self.lhs.size()[0], self.lhs.size()[1], self.lhs.size()[1]))
        else:
            return torch.Size((self.lhs.size()[0], self.lhs.size()[0]))

    def _transpose_nonbatch(self):
        return MatmulLazyVariable(self.rhs.transpose(-1, -2), self.lhs.transpose(-1, -2))

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        left_vals = self.lhs[batch_indices, left_indices, :]
        right_vals = self.rhs[batch_indices, :, right_indices]
        return (left_vals * right_vals).sum(-1)

    def _get_indices(self, left_indices, right_indices):
        res = self.lhs.index_select(-2, left_indices) * self.rhs.index_select(-1, right_indices).transpose(-1, -2)
        return res.sum(-1)

    def diag(self):
        return (self.lhs * self.rhs.transpose(-1, -2)).sum(-1)

    def evaluate(self):
        return torch.matmul(self.lhs, self.rhs)
