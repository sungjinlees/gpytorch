from lazy_function import LazyFunction
from lazy_variable import LazyVariable

from kron import Kron, KronVariable


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
    if not issubclass(function_class, LazyFunction):
        raise RuntimeError('%s must be a LazyFunction subclass' % function_class.__name__)
    if not hasattr(cls, '_lazy_functions'):
        cls._lazy_functions = {}
    cls._lazy_functions[_ClassList(input_classes)] = function_class
