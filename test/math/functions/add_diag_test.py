import math
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch.lazy import Kron, KronVariable
from gpytorch.math.functions import AddDiag

def test_forward():
    a = Parameter(torch.Tensor([5])) 
    b = Variable(torch.ones(3, 3)) 
    output = AddDiag()(b, a)

    actual = torch.Tensor([
        [6, 1, 1],
        [1, 6, 1],
        [1, 1, 6],
    ])
    assert(torch.norm(output.data - actual) < 1e-7)


def test_backward():
    grad = torch.randn(3, 3)

    a = Parameter(torch.Tensor([3])) 
    b = Variable(torch.ones(3, 3), requires_grad=True) 
    output = AddDiag()(b, a)
    output.backward(gradient=grad)

    assert(math.fabs(a.grad.data[0] - grad.trace()) < 1e-6)
    assert(torch.norm(b.grad.data - grad) < 1e-6)


def test_forward_on_kron():
    def evaluate_stub(self, *inputs, **kwargs):
        raise Exception('Forward should not be called')

    a = Variable(torch.randn(3, 3)) 
    b = Variable(torch.ones(3, 3)) 
    c = Parameter(torch.Tensor([0])) 
    d = Parameter(torch.Tensor([5])) 

    kron_var = Kron()(a, b, c)
    kron_var.evaluate = evaluate_stub.__get__(kron_var, kron_var.__class__)
    res = AddDiag()(kron_var, d)
    assert(isinstance(res, KronVariable))
    assert(res.inputs[0:2] == (a, b))
    assert(res.inputs[2].data.squeeze()[0] == 5)

