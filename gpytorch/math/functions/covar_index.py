from operator import mul
import torch
from torch.autograd import Function
from gpytorch.lazy import Kron, KronVariable


prod = lambda iterable: reduce(mul, iterable, 1)


class CovarIndex(Function):
    def __init__(self, size, idx1, idx2=None):
        self.idx1 = []
        if not hasattr(idx1, '__iter__'):
            idx1 = idx1,
        for i in range(len(size)):
            if i < len(idx1):
                if isinstance(idx1[i], int):
                    self.idx1.append(slice(idx1[i], idx1[i] + 1, None))
                else:
                    self.idx1.append(idx1[i])
            else:
                self.idx1.append(slice(None, None, None))

        self.idx2 = []
        if not hasattr(idx2, '__iter__'):
            idx2 = idx2,
        for i in range(len(size)):
            if i < len(idx2):
                if isinstance(idx2[i], int):
                    self.idx2.append(slice(idx2[i], idx2[i] + 1, None))
                else:
                    self.idx2.append(idx2[i])
            else:
                self.idx2.append(slice(None, None, None))

        self.size = tuple(size)
        self.idx1 = tuple(self.idx1)
        self.idx2 = tuple(self.idx2)
        self.full_idx = (slice(None, None, None),) * len(self.size)

    
    def forward(self, covar_matrix):
        if not len(self.idx1) and not len(self.idx2):
            return covar_matrix

        total_size = prod(self.size)
        if total_size != covar_matrix.size(0) or total_size != covar_matrix.size(1):
            raise RuntimeError('Mismatched dimensions')

        covar_matrix = covar_matrix.view(*[self.size + self.size])
        covar_matrix = covar_matrix[self.idx1 + self.idx2]

        self.new_size1 = tuple(covar_matrix.size())[0:len(self.size)]
        self.new_size2 = tuple(covar_matrix.size())[len(self.size):]
        covar_matrix = covar_matrix.contiguous().view(prod(self.new_size1), prod(self.new_size2))
        print('Inefficient', covar_matrix.size())
        return covar_matrix


    def backward(self, grad_output):
        if not len(self.idx1) and not len(self.idx2):
            return grad_output

        grad_output = grad_output.view(*[self.new_size1 + self.new_size2])
        grad_input = grad_output.new().resize_(self.size * 2).zero_()
        intermediate = grad_input.new().resize_(self.new_size1 + self.size).zero_()
        intermediate[self.full_idx + self.idx2].copy_(grad_output)
        grad_input[self.idx1] = intermediate
        final_size = prod(self.size)
        grad_input = grad_input.view(final_size, final_size)
        return grad_input, None


class KronCovarIndex(Kron):
    def __call__(self, mat_a, mat_b, diag, **kwargs):
        print('Should be', CovarIndex(self.size, self.idx1, self.idx2)(Kron()(mat_a, mat_b, diag).evaluate()).size())
        mat_a = CovarIndex(self.size[:1], self.idx1[:1], self.idx2[:1])(mat_a)
        mat_b = CovarIndex(self.size[1:], self.idx1[1:], self.idx2[1:])(mat_b)
        res = Kron()(mat_a, mat_b, diag)
        print('Efficient', (res + 0).size())
        return res


CovarIndex.register_lazy_function((KronVariable,), KronCovarIndex)
