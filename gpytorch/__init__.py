from distribution import Distribution
from observation_model import ObservationModel

import torch

# Monkey-patch the call method
orig_call = torch.autograd.Function.__call__

def new_call(ctx, *inputs, **kwargs):
    for input in inputs:
        if hasattr(input, 'evaluate'):
            input.evaluate()
    return orig_call(ctx, *inputs, **kwargs)

torch.autograd.Function.__call__ = new_call
