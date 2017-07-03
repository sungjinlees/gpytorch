import torch
from torch.autograd import Variable

class LazyVariable(Variable):
    def check_inputs(self, inputs):
        pass


    @property
    def inputs(self):
        return self.__inputs


    @inputs.setter
    def inputs(self, inputs):
        self.check_inputs(inputs)
        self.__inputs = inputs

