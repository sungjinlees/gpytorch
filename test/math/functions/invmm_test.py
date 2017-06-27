import math
import torch
from torch.autograd import Variable
from gpytorch.lazy import Kron
from gpytorch.math.functions import Invmm 


def test_forward():
    for n_cols in [2, 3, 4]:
        a = torch.Tensor([
            [5, -3, 0],
            [-3, 5, 0],
            [0, 0, 2],
        ])
        b = torch.randn(3, n_cols)
        actual = a.inverse().mm(b)

        a_var = Variable(a)
        b_var = Variable(b)
        out_var = Invmm()(a_var, b_var)
        res = out_var.data

        assert(torch.norm(actual - res) < 1e-4)


def test_forward_for_kron():
    def evaluate_stub(self):
        raise Exception('Should not call evaluate')

    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.Tensor([
        [3, 0, -3],
        [0, 2, 1],
        [-3, 1, 6],
    ])
    diag = torch.Tensor([3])
    other = torch.randn(9, 3)

    a_var = Variable(a)
    b_var = Variable(b)
    diag_var = Variable(diag)
    other_var = Variable(other)

    kron_var = Kron()(a_var, b_var, diag_var)
    actual = Invmm()(kron_var.evaluate(), other_var)

    kron_var = Kron()(a_var, b_var, diag_var)
    kron_var.evaluate = evaluate_stub.__get__(kron_var, kron_var.__class__)
    res = Invmm()(kron_var, other_var)

    assert(torch.norm(actual.data - res.data) < 1e-4)


def test_backward():
    for n_cols in [2, 3, 4]:
        a = torch.Tensor([
            [5, -3, 0],
            [-3, 5, 0],
            [0, 0, 2],
        ])
        b = torch.ones(3, 3).fill_(2)
        c = torch.randn(3, n_cols)
        actual_a_grad = -torch.mm(
            a.inverse().mul_(0.5).mm(torch.eye(3, n_cols)),
            a.inverse().mul_(0.5).mm(c).t()
        ) * 2 * 2
        actual_c_grad = (a.inverse() / 2).t().mm(torch.eye(3, n_cols)) * 2

        a_var = Variable(a, requires_grad=True)
        c_var = Variable(c, requires_grad=True)
        out_var = a_var.mul(Variable(b))
        out_var = Invmm()(out_var, c_var) 
        out_var = out_var.mul(Variable(torch.eye(3, n_cols))).sum() * 2
        out_var.backward()
        a_res = a_var.grad.data
        c_res = c_var.grad.data

        assert(torch.norm(actual_a_grad - a_res) < 1e-4)
        assert(torch.norm(actual_c_grad - c_res) < 1e-4)


def test_backward_for_kron():
    def evaluate_stub(self):
        raise Exception('Should not call evaluate')

    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.Tensor([
        [3, 0, -3],
        [0, 2, 1],
        [-3, 1, 6],
    ])
    diag = torch.Tensor([3])
    other = torch.randn(9, 3)

    a_var = Variable(a, requires_grad=True)
    b_var = Variable(b, requires_grad=True)
    diag_var = Variable(diag, requires_grad=True)
    other_var = Variable(other, requires_grad=True)
    kron_var = Kron()(a_var, b_var, diag_var)
    actual = Invmm()(kron_var.evaluate(), other_var)
    actual.sum().backward()

    grad_a_actual = a_var.grad.data
    grad_b_actual = b_var.grad.data
    grad_diag_actual = diag_var.grad.data
    grad_other_actual = other_var.grad.data

    a_var = Variable(a, requires_grad=True)
    b_var = Variable(b, requires_grad=True)
    diag_var = Variable(diag, requires_grad=True)
    other_var = Variable(other, requires_grad=True)
    kron_var = Kron()(a_var, b_var, diag_var)
    kron_var.evaluate = evaluate_stub.__get__(kron_var, kron_var.__class__)
    res = Invmm()(kron_var, other_var)
    res.sum().backward()

    assert(torch.norm(grad_other_actual - other_var.grad.data) < 1e-4)
    assert(torch.norm(grad_diag_actual - diag_var.grad.data) < 1e-4)
    assert(torch.norm(grad_a_actual - a_var.grad.data) < 1e-4)
    assert(torch.norm(grad_b_actual - b_var.grad.data) < 1e-4)
