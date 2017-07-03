import torch
from torch.autograd import Variable
from .lazy import LazyVariable, LazyFunction


class ClassPropertyDescriptor(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)


# Wrapper class for list of input classes
class _ClassList(tuple):
    def matches(self, necessary_class_list):
        if len(self) != len(necessary_class_list):
            return False

        for self_class, necessary_class in zip(self, necessary_class_list):
            if not issubclass(self_class, necessary_class):
                return False

        return True


    def __le__(self, other):
        if len(self) == len(other):
            return len(self) < len(other)
        self_non_variable_classes = [cls for cls in self if cls != Variable]
        other_non_variable_classes = [cls for cls in other if cls != Variable]
        return len(self_non_variable_classes) <= len(other_non_variable_classes)


    def __lt__(self, other):
        return self.__lt__(other) and not self.__eq__(other)
        

    def __ge__(self, other):
        return other.__le__(self)
        

    def __gt__(self, other):
        return other.__lt__(self)



# Registration mechanism for efficient function calculations
def register_lazy_function(cls, input_classes, function_class):
    if function_class.__name__ not in [subcls.__name__ for subcls in cls.lazy_functions.values()]:
        cls.lazy_functions[_ClassList(input_classes)] = function_class


# New method to get function's lazy versions
@classproperty
def lazy_functions(cls):
    if not hasattr(cls, '_lazy_functions_%s' % cls.__name__):
        setattr(cls, '_lazy_functions_%s' % cls.__name__, {})
    return getattr(cls, '_lazy_functions_%s' % cls.__name__)


# Registration mechanism for efficient function calculations
@classmethod
def register_lazy_function(cls, input_classes, function_class):
    cls.lazy_functions[_ClassList(input_classes)] = function_class


# Monkey-patch the call method
orig_call = torch.autograd.Function.__call__
def new_call(ctx, *inputs, **kwargs):
    actual_input_classes = _ClassList([input.__class__ for input in inputs])

    # The function we will call, and the classes that our function is intended for
    needed_input_classes = [Variable] * len(inputs)
    call = orig_call.__get__(ctx, ctx.__class__)

    # Special action if we have any LazyVariables
    # See if there are any registered lazy functions that exploit the structure
    # of the lazy variables
    if len(set(actual_input_classes) - {Variable}):
        function_matches = sorted([(input_classes, function_class)
                for input_classes, function_class in ctx.lazy_functions.items()
                if actual_input_classes.matches(input_classes)])
        if len(function_matches):
            needed_input_classes, function_class = function_matches[0]
            cls = ctx.__class__
            ctx.__class__ = cls.__class__(function_class.__name__, (function_class,), {})
            call = ctx.__call__

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
    return call(*inputs, **kwargs)


torch.autograd.Function.lazy_functions = lazy_functions
torch.autograd.Function.register_lazy_function = register_lazy_function
torch.autograd.Function.__call__ = new_call



from distribution import Distribution
from observation_model import ObservationModel

