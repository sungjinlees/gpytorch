from distribution import Distribution
from observation_model import ObservationModel

import torch
from torch.autograd import Variable
from gpytorch.lazy import LazyVariable, LazyFunction, _ClassList


# New method to get function's lazy versions
def lazy_functions(self):
    if hasattr(self, '_lazy_functions'):
        return self._lazy_functions
    return dict()


# Monkey-patch the call method
orig_call = torch.autograd.Function.__call__
def new_call(ctx, *inputs, **kwargs):
    actual_input_classes = _ClassList([input.__class__ for input in inputs])

    # The function we will call, and the classes that our function is intended for
    needed_input_classes = [Variable] * len(inputs)
    function = ctx

    # Special action if we have any LazyVariables
    # See if there are any registered lazy functions that exploit the structure
    # of the lazy variables
    if len(set(actual_input_classes) - {Variable}):
        function_matches = sorted([(input_classes, function_class)
                for input_classes, function_class in ctx.lazy_functions().items()
                if actual_input_classes.matches(input_classes)])
        if len(function_matches):
            needed_input_classes, function_class = function_matches[0]
            function = function_class()

        # Evaluate any lazy variables that we cannot exploit the structure of
        for input, needed_input_class in zip(inputs, needed_input_classes):
            if isinstance(input, LazyVariable) and not(issubclass(needed_input_class, LazyVariable)):
                input.evaluate()

        # Get a flattened list of inputs
        inputs = [input_component for input in inputs
                for input_component in (
                    input.inputs if isinstance(input, LazyVariable) else [input]
                )]

    # Perform the call method
    if function == ctx:
        return orig_call(ctx, *inputs, **kwargs)
    else:
        return function.__call__(*inputs, **kwargs)


torch.autograd.Function.lazy_functions = lazy_functions
torch.autograd.Function.__call__ = new_call
