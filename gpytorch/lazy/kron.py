import torch
from torch.autograd import Variable
from gpytorch.lazy import LazyFunction, LazyVariable


class KronVariable(LazyVariable):
    pass


class Kron(LazyFunction):
    variable_class = KronVariable


    def forward(self, mat_1, mat_2, diag):
        if diag.numel() == 1:
            self.diag_val = diag.squeeze()[0]
        else:
            raise RuntimeError('Diag must be a single-element tensor')

        self.save_for_backward(mat_1, mat_2)

        m, n = mat_1.size()
        p, q = mat_2.size()

        mat_1_exp = mat_1.unsqueeze(1).unsqueeze(3).expand(m, p, n, q)
        mat_2_exp = mat_2.unsqueeze(0).unsqueeze(2).expand(m, p, n, q)
        res = (mat_1_exp * mat_2_exp).view(m * p, n * q)
        if self.diag_val is not None:
            res.add_(torch.eye(m * p, n * q).mul_(self.diag_val))

        return res


    def backward(self, grad_output):
        grad_mat_1 = None
        grad_mat_2 = None
        diag_grad = None
        mat_1, mat_2 = self.saved_tensors

        if self.needs_input_grad[2]:
            diag_grad = grad_output.new().resize_(1)
            diag_grad.fill_(grad_output.trace())

        m, n = mat_1.size()
        p, q = mat_2.size()
        grad_output = grad_output.view(m, p, n, q)

        if self.needs_input_grad[0]:
            mat_2_exp = mat_2.unsqueeze(0).unsqueeze(2).expand(m, p, n, q)
            grad_mat_1 = torch.mul(mat_2_exp, grad_output).sum(3).sum(1).squeeze(3).squeeze(1)

        if self.needs_input_grad[1]:
            mat_1_exp = mat_1.unsqueeze(1).unsqueeze(3).expand(m, p, n, q)
            grad_mat_2 = torch.mul(mat_1_exp, grad_output).sum(2).sum(0).squeeze(2).squeeze(0)

        return grad_mat_1, grad_mat_2, diag_grad
