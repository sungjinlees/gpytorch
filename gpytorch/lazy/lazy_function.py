import torch
from torch.autograd import Function

class LazyFunction(Function):
    '''
    A lazy-evaluated function.
    Useful when variable data can be represented by a more efficient representation.
    (e.g. Kronecker, Toeplitz, etc.)
    '''

    def save_for_backward(self, *args):
        return super(LazyFunction, self).save_for_backward(*args)


    @property
    def saved_tensors(self):
        return self.to_save


    def lazy_forward(self, *inputs):
        return inputs[0].new()


    def __call__(self, *inputs, **kwargs):
        # Evaluate with a lazy forward method
        orig_forward = self.forward
        self.forward = self.lazy_forward
        res = super(LazyFunction, self).__call__(*inputs, **kwargs)
        self.forward = orig_forward

        # Store the inputs and function class for later
        res.inputs = inputs
        res.lazy_type = self.__class__

        # Provide a method for evaluation
        def evaluate(self):
            inputs = [input.data for input in self.inputs]
            res = orig_forward(*inputs)
            self.data = res
            return self

        res.evaluate = evaluate.__get__(res, res.__class__)
        return res
