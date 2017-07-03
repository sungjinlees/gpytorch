import torch
from torch.autograd import Variable
from gpytorch.lazy import Kron
from gpytorch.math.functions import CovarIndex


a = torch.Tensor([
    [1, 4, 5],
    [4, 2, 6],
    [5, 6, 3],
])
b = torch.Tensor([
    [10, 20],
    [20, 10],
])
kron = torch.Tensor([
    [ 10,   20,   40,  80,   50,  100],
    [ 20,   10,   80,  40,  100,   50],
    [ 40,   80,   20,  40,   60,  120],
    [ 80,   40,   40,  20,  120,   60],
    [ 50,  100,   60, 120,   30,   60],
    [100,   50,  120,  60,   60,   30],
])


def test_forward_with_one_index():
    actual = torch.Tensor([
        [ 60, 120,],
        [120,  60,],
    ])
    output = Variable(kron, requires_grad=True)
    res = CovarIndex(torch.Size((3, 2)), 2, 1)(output).data
    assert(torch.norm(res - actual) < 1e-4)


def test_forward_with_two_indices():
    actual = torch.Tensor([
        [ 10,   40,   50,],
        [ 20,   80,  100,],
        [ 40,   20,   60,],
        [ 80,   40,  120,],
        [ 50,   60,   30,],
        [100,  120,   60,],
    ])
    output = Variable(kron, requires_grad=True)
    idx1 = (slice(None, None, None), slice(None, None, None))
    idx2 = (slice(None, None, None), 0)
    res = CovarIndex(torch.Size((3, 2)), idx1, idx2)(output).data
    assert(torch.norm(res - actual) < 1e-4)


def test_backward_with_one_index():
    grad = torch.Tensor([
        [ 1,   2,],
        [ 3,   4,],
    ])
    actual = torch.Tensor([
        [  0,    0,    0,   0,    0,    0],
        [  0,    0,    0,   0,    0,    0],
        [  0,    0,    0,   0,    0,    0],
        [  0,    0,    0,   0,    0,    0],
        [  0,    0,    1,   2,    0,    0],
        [  0,    0,    3,   4,    0,    0],
    ])

    output = Variable(kron, requires_grad=True)
    CovarIndex(torch.Size((3, 2)), 2, 1)(output).backward(gradient=grad)
    assert(torch.norm(output.grad.data - actual) < 1e-4)


def test_backward_with_two_indices():
    grad = torch.Tensor([
        [ 1,   2,   3,],
        [ 4,   5,   6,],
        [ 7,   8,   9,],
        [10,  11,  12,],
        [13,  14,  15,],
        [16,  17,  18,],
    ])
    actual = torch.Tensor([
        [  1,    0,    2,   0,    3,    0],
        [  4,    0,    5,   0,    6,    0],
        [  7,    0,    8,   0,    9,    0],
        [ 10,    0,   11,   0,   12,    0],
        [ 13,    0,   14,   0,   15,    0],
        [ 16,    0,   17,   0,   18,    0],
    ])

    output = Variable(kron, requires_grad=True)
    idx1 = (slice(None, None, None), slice(None, None, None))
    idx2 = (slice(None, None, None), 0)
    CovarIndex(torch.Size((3, 2)), idx1, idx2)(output).backward(gradient=grad)
    assert(torch.norm(output.grad.data - actual) < 1e-4)
