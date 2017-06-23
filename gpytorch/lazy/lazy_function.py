import torch
from torch.autograd import Function
from .lazy_variable import LazyVariable

class LazyFunction(Function):
    '''
    A lazy-evaluated function.
    Useful when variable data can be represented by a more efficient representation.
    (e.g. Kronecker, Toeplitz, etc.)
    '''
    variable_class = LazyVariable


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

        # Change class of result
        cls = res.__class__
        res._original_class = cls
        res.__class__ = cls.__class__(self.variable_class.__name__, (self.variable_class,), {})

        # Provide a method for evaluation
        def evaluate(self):
            inputs = [input.data for input in self.inputs]
            res = orig_forward(*inputs)
            self.data = res
            
            # After evaluation, variable becomes a normal variable again
            self.__class__ = self._original_class
            del self.evaluate
            return self

        res.evaluate = evaluate.__get__(res, res.__class__)
        return res
