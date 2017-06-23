import torch
from torch.autograd import Variable, Function
from gpytorch.lazy import LazyFunction

class TestFunction(LazyFunction):
    def forward(self, input):
        return input + 2


def test_lazy_evaluation_does_not_occur_on_call():
    def forward_stub(self, input):
        raise Exception('Should not have been called')

    func = TestFunction()
    func.forward = forward_stub.__get__(func, func.__class__)

    a_var = Variable(torch.ones(3, 2) * 2)
    func(a_var)


def test_lazy_evaluation_occurs_on_evaluate():
    func = TestFunction()
    a_var = Variable(torch.ones(3, 2) * 2)
    res = func(a_var)
    res.evaluate()
    assert(torch.norm(res.data - torch.ones(3, 2) * 4) < 1e-5)


def test_lazy_evaluation_occurs_on_next_call():
    func = TestFunction()
    a_var = Variable(torch.ones(3, 2) * 2)
    res = 2 + func(a_var)
    assert(torch.norm(res.data - torch.ones(3, 2) * 6) < 1e-5)
