import math
import torch
from torch.autograd import Function, Variable
from torch.nn import Parameter
from gpytorch.lazy import Kron, KronVariable

a = torch.Tensor([
    [1, 2, 3],
    [4, 5, 6]
])
b = torch.Tensor([
    [10],
    [20],
    [30],
    [40],
])

def test_forward_without_diag():
    actual = torch.Tensor([
        [ 10,   20,   30],
        [ 20,   40,   60],
        [ 30,   60,   90],
        [ 40,   80,  120],
        [ 40,   50,   60],
        [ 80,  100,  120],
        [120,  150,  180],
        [160,  200,  240],
    ])
    cvar = Parameter(torch.Tensor([0]))
    output = Kron()(Variable(a), Variable(b), cvar)
    assert(isinstance(output, KronVariable))

    res = output.evaluate().data
    assert(torch.norm(res - actual) < 1e-4)


def test_forward_with_diag():
    actual = torch.Tensor([
        [ 13,   20,   30],
        [ 20,   43,   60],
        [ 30,   60,   93],
        [ 40,   80,  120],
        [ 40,   50,   60],
        [ 80,  100,  120],
        [120,  150,  180],
        [160,  200,  240],
    ])
    output = Kron()(Variable(a), Variable(b), Parameter(torch.Tensor([3])))
    assert(isinstance(output, KronVariable))

    res = output.evaluate().data
    assert(torch.norm(res - actual) < 1e-4)


def test_backward_without_diag():
    gradient = torch.Tensor([
        [ 1,   2,   3],
        [ 4,   5,   6],
        [ 7,   8,   9],
        [10,  11,  12],
        [13,  14,  15],
        [16,  17,  18],
        [19,  20,  21],
        [22,  23,  24],
    ])

    actual_a = 10 * torch.Tensor([
        [ 1,   2,   3],
        [13,  14,  15],
    ]) + 20 * torch.Tensor([
        [ 4,   5,   6],
        [16,  17,  18],
    ]) + 30 * torch.Tensor([
        [ 7,   8,   9],
        [19,  20,  21],
    ]) + 40 * torch.Tensor([
        [10,  11,  12],
        [22,  23,  24],
    ])

    actual_b = torch.Tensor([
        [1], [4], [7], [10]
    ]) + 2 * torch.Tensor([
        [2], [5], [8], [11]
    ]) + 3 * torch.Tensor([
        [3], [6], [9], [12]
    ]) + 4 * torch.Tensor([
        [13], [16], [19], [22]
    ]) + 5 * torch.Tensor([
        [14], [17], [20], [23]
    ]) + 6 * torch.Tensor([
        [15], [18], [21], [24]
    ])

    avar = Variable(a, requires_grad=True)
    bvar = Variable(b, requires_grad=True)
    cvar = Parameter(torch.Tensor([0]))
    outvar = Kron()(avar, bvar, cvar).evaluate()
    outvar.backward(gradient=gradient)

    assert(torch.norm(avar.grad.data - actual_a) < 1e-6)
    assert(torch.norm(bvar.grad.data - actual_b) < 1e-6)


def test_backward_with_diag():
    gradient = torch.Tensor([
        [ 1,   2,   3],
        [ 4,   5,   6],
        [ 7,   8,   9],
        [10,  11,  12],
        [13,  14,  15],
        [16,  17,  18],
        [19,  20,  21],
        [22,  23,  24],
    ])

    actual_a = 10 * torch.Tensor([
        [ 1,   2,   3],
        [13,  14,  15],
    ]) + 20 * torch.Tensor([
        [ 4,   5,   6],
        [16,  17,  18],
    ]) + 30 * torch.Tensor([
        [ 7,   8,   9],
        [19,  20,  21],
    ]) + 40 * torch.Tensor([
        [10,  11,  12],
        [22,  23,  24],
    ])

    actual_b = torch.Tensor([
        [1], [4], [7], [10]
    ]) + 2 * torch.Tensor([
        [2], [5], [8], [11]
    ]) + 3 * torch.Tensor([
        [3], [6], [9], [12]
    ]) + 4 * torch.Tensor([
        [13], [16], [19], [22]
    ]) + 5 * torch.Tensor([
        [14], [17], [20], [23]
    ]) + 6 * torch.Tensor([
        [15], [18], [21], [24]
    ])

    avar = Variable(a, requires_grad=True)
    bvar = Variable(b, requires_grad=True)
    cvar = Parameter(torch.Tensor([3]))
    outvar = Kron()(avar, bvar, cvar).evaluate()
    outvar.backward(gradient=gradient)

    assert(torch.norm(avar.grad.data - actual_a) < 1e-6)
    assert(torch.norm(bvar.grad.data - actual_b) < 1e-6)
    assert(math.fabs(cvar.grad.data[0] - torch.trace(gradient)) < 1e-6)

