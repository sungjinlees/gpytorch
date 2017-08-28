from ..utils import function_factory
from ..utils import circulant
from .lazy_variable import LazyVariable
import pdb

class CirculantGridPreconditionedLazyVariable(LazyVariable):
    def __init__(self, circulant_column, lazy_variable, mul_constant=1):
        self.lazy_variable = lazy_variable
        self.mul_constant = mul_constant
        self.circulant_column = circulant_column

    def preconditioner_matmul(self, rhs):
        if rhs.ndimension() == 1:
            return (1 / self.mul_constant) * circulant.circulant_invmv(self.circulant_column, rhs)
        else:
            return (1 / self.mul_constant) * circulant.circulant_invmm(self.circulant_column, rhs)

    def _mm_closure_factory(self, *args):
        return self.lazy_variable._mm_closure_factory(*args)

    def _derivative_quadratic_form_factory(self, *args):
        return self.lazy_variable._derivative_quadratic_form_factory(*args)

    def diag(self):
        return self.lazy_variable.diag()

    def add_diag(self, diag):
        return self.lazy_variable.add_diag(diag)

    def add_jitter(self):
        jittered_lazy_variable = self.lazy_variable.add_jitter()
        return CirculantGridPreconditionedLazyVariable(self.circulant_column, jittered_lazy_variable)

    def evaluate(self):
        return self.lazy_variable.evaluate()

    def gp_marginal_log_likelihood(self, target):
        return self.lazy_variable.gp_marginal_log_likelihood(target)

    def invmm(self, rhs_mat):
        return self.lazy_variable.invmm(rhs_mat)

    def mm(self, rhs_mat):
        return self.lazy_variable.mm(rhs_mat)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar, num_samples):
        return self.lazy_variable.monte_carlo_log_likelihood(log_probability_func,
                                                             train_y,
                                                             variational_mean,
                                                             chol_var_covar,
                                                             num_samples)

    def mul(self, constant):
        new_lazy_variable = self.lazy_variable.mul(constant)
        return CirculantGridPreconditionedLazyVariable(self.circulant_column, new_lazy_variable, constant.data)

    def mul_(self, constant):
        self.lazy_variable.mul_(constant)
        self.circulant_column.mul_(constant)

    def representation(self, *args):
        """
        Returns the variables that are used to define the LazyVariable
        """
        return self.circulant_column, self.lazy_variable.representation()

    def trace_log_det_quad_form(self, mu_diffs, chol_covar_1, num_samples=10):
        if not hasattr(self, '_trace_log_det_quad_form_class'):
            tlqf_function_factory = function_factory.trace_logdet_quad_form_factory
            self._trace_log_det_quad_form_class = tlqf_function_factory(self._mm_closure_factory,
                                                                        self._derivative_quadratic_form_factory)

        #self.circulant_column = circulant.frobenius_circulant_approximation_toeplitz(self.lazy_variable.c.data)
        precondition_closure = self.preconditioner_matmul
        covar2_args = self.lazy_variable.variational_representation()
        return self._trace_log_det_quad_form_class(num_samples, precondition_closure)(mu_diffs,
                                                                                      chol_covar_1,
                                                                                      *covar2_args)

    def exact_posterior_alpha(self, train_mean, train_y):
        return self.lazy_variable.exact_posterior_alpha(train_mean, train_y)

    def exact_posterior_mean(self, test_mean, alpha):
        return self.lazy_variable.exact_posterior_mean(test_mean, alpha)

    def variational_posterior_mean(self, alpha):
        return self.lazy_variable.variational_posterior_mean(alpha)

    def variational_posterior_covar(self):
        return self.lazy_variable

    def __getitem__(self, index):
        return self.lazy_variable.__getitem__(index)
