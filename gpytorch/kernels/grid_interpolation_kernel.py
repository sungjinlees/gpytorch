import torch
from torch.autograd import Variable
from .kernel import Kernel
from gpytorch.utils.interpolation import Interpolation
from gpytorch.utils import circulant
from gpytorch.lazy import ToeplitzLazyVariable, CirculantGridPreconditionedLazyVariable
import pdb

class GridInterpolationKernel(Kernel):
    def __init__(self, base_kernel_module, whittle_precondition_factor=0):
        super(GridInterpolationKernel, self).__init__()
        self.base_kernel_module = base_kernel_module
        self.whittle_precondition_factor = whittle_precondition_factor
        self.grid = None

    def initialize_interpolation_grid(self, grid_size, grid_bounds):
        super(GridInterpolationKernel, self).initialize_interpolation_grid(grid_size, grid_bounds)
        grid_size = grid_size
        grid = torch.linspace(grid_bounds[0], grid_bounds[1], grid_size - 2)

        grid_diff = grid[1] - grid[0]

        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        actual_lower = grid_bounds[0] - grid_diff
        actual_upper = grid_bounds[1] + grid_diff
        self.grid = Variable(torch.linspace(actual_lower,
                                            actual_upper,
                                            grid_size))

        if self.whittle_precondition_factor > 0:
            extended_grid_size = grid_size + grid_size * self.whittle_precondition_factor * 2
            lower_extended_bound = actual_lower - self.whittle_precondition_factor * grid_size * grid_diff
            upper_extended_bound = actual_upper + self.whittle_precondition_factor * grid_size * grid_diff
            self.extended_grid = Variable(torch.linspace(lower_extended_bound,
                                                         upper_extended_bound,
                                                         extended_grid_size))
            self.extended_grid_size = extended_grid_size

        return self

    def forward(self, x1, x2, **kwargs):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            raise RuntimeError(' '.join([
                'The grid interpolation kernel can only be applied to inputs of a single dimension at this time \
                until Kronecker structure is implemented.'
            ]))

        if self.grid is None:
            raise RuntimeError(' '.join([
                'This GridInterpolationKernel has no grid. Call initialize_interpolation_grid \
                 on a GPModel first.'
            ]))

        both_min = torch.min(x1.min(0)[0].data, x2.min(0)[0].data)[0]
        both_max = torch.max(x1.max(0)[0].data, x2.max(0)[0].data)[0]

        if both_min < self.grid_bounds[0] or both_max > self.grid_bounds[1]:
            # Out of bounds data is still ok if we are specifically computing kernel values for grid entries.
            if torch.abs(both_min - self.grid[0].data)[0] > 1e-7 or torch.abs(both_max - self.grid[-1].data)[0] > 1e-7:
                raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                    Grid bounds were ({}, {}), but min = {}, max = {}'.format(self.grid_bounds[0],
                                                                                              self.grid_bounds[1],
                                                                                              both_min,
                                                                                              both_max))

        J1, C1 = Interpolation().interpolate(self.grid.data, x1.data.squeeze())
        J2, C2 = Interpolation().interpolate(self.grid.data, x2.data.squeeze())

        k_UU = self.base_kernel_module(self.grid[0], self.grid, **kwargs).squeeze()
        K_XX = ToeplitzLazyVariable(k_UU, J1, C1, J2, C2)

        if self.whittle_precondition_factor > 0:
            grid_start = self.whittle_precondition_factor * self.grid_size
            k_whittle = self.base_kernel_module(self.extended_grid,
                                                self.extended_grid[grid_start],
                                                **kwargs).squeeze().data

            circ_whittle = k_UU.data.new().resize_as_(k_UU.data).fill_(0)

            for i in range(len(circ_whittle)):
                index = torch.LongTensor(list(range(i, self.extended_grid_size, self.grid_size)))
                circ_whittle[i] = k_whittle.index_select(0, index).sum()

            circ_whittle[0] = circ_whittle[0] + 1e-4
            circ_frobenius = circulant.frobenius_circulant_approximation_toeplitz(k_UU.data)

            # pdb.set_trace()

            K_XX = CirculantGridPreconditionedLazyVariable(circ_whittle, K_XX)

        return K_XX
