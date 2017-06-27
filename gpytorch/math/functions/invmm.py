import torch
from torch.autograd import Function, Variable
from gpytorch.utils import pd_catcher
from gpytorch.utils.kron import kron_forward, kron_backward
from gpytorch.lazy import Kron, KronVariable, LazyVariable

# Returns input_1^{-1} input_2
class Invmm(Function):
    def forward(self, chol_matrix, input_2):
        res = input_2.potrs(chol_matrix)
        self.save_for_backward(chol_matrix, input_2, res)
        return res


    def backward(self, grad_output):
        chol_matrix, input_2, input_1_t_input_2 = self.saved_tensors
        grad_input_1 = None
        grad_input_2 = None

        # input_1 gradient
        if self.needs_input_grad[0]:
            grad_input_1 = torch.mm(grad_output, input_1_t_input_2.t())
            grad_input_1 = grad_input_1.potrs(chol_matrix, out=grad_input_1)
            grad_input_1 = grad_input_1.mul_(-1)

        # input_2 gradient
        if self.needs_input_grad[1]:
            grad_input_2 = grad_output.potrs(chol_matrix)

        return grad_input_1, grad_input_2


    def __call__(self, input_1_var, input_2_var):
        if isinstance(input_1_var, LazyVariable):
            return super(Invmm, self).__call__(input_1_var, input_2_var)

        # If there is no structure to exploit, do a Cholesky decomposition
        if not hasattr(input_1_var, 'chol_data'):
            def add_jitter():
                print('Matrix not positive definite. Adding jitter:')
                input_1_var.add_(Variable(torch.eye(*input_1_var.size()) * 1e-5))
                return False

            @pd_catcher(catch_function=add_jitter)
            def chol_data_closure():
                input_1_var.chol_data = input_1_var.data.potrf()
                return True

            has_completed = False
            while not has_completed:
                has_completed = chol_data_closure()

        # Switch the variable data with cholesky data, for computation
        orig_data = input_1_var.data
        input_1_var.data = input_1_var.chol_data
        res = super(Invmm, self).__call__(input_1_var, input_2_var)

        # Revert back to original data
        input_1_var.data = orig_data
        return res


class KronInvmm(Function):
    def forward(self, mat_a_eig_data, mat_b_eig_data, diag, other):
        mat_a_evec = mat_a_eig_data[:, :-1]
        mat_b_evec = mat_b_eig_data[:, :-1]
        mat_a_eval = mat_a_eig_data[:, -1]
        mat_b_eval = mat_b_eig_data[:, -1]
        diag_val = diag.squeeze()[0]

        m = mat_a_evec.size(0)
        p = mat_b_evec.size(0)

        # Caculate evec matrix
        evec = kron_forward(mat_a_evec, mat_b_evec)

        # Eval with noise
        eval_with_noise = torch.ger(mat_a_eval, mat_b_eval).view(-1).add_(diag_val)

        # Result
        evec_eval_inv_prod = evec / eval_with_noise.unsqueeze(0).expand(m * p, m * p)
        res = evec_eval_inv_prod.mm(evec.t().mm(other))
        self.evec = evec
        self.evec_eval_inv_prod = evec_eval_inv_prod
        self.save_for_backward(res)
        return res


    def backward(self, grad_output):
        evec = self.evec
        evec_eval_inv_prod = self.evec_eval_inv_prod
        res, = self.saved_tensors

        mat_a_grad = None
        mat_b_grad = None
        diag_grad = None
        other_grad = None

        if any(self.needs_input_grad[:3]):
            kron_grad = torch.mm(grad_output, res.t())
            kron_grad = torch.mm(evec_eval_inv_prod, evec.t().mm(kron_grad))
            kron_grad.mul_(-1)
            mat_a_grad, mat_b_grad, diag_grad = kron_backward(
                    self.mat_a,
                    self.mat_b,
                    kron_grad,
                    self.needs_input_grad[:3]
            )
        
        if self.needs_input_grad[3]:
            other_grad = torch.mm(evec_eval_inv_prod, evec.t().mm(grad_output))

        return mat_a_grad, mat_b_grad, diag_grad, other_grad



    def __call__(self, mat_a_var, mat_b_var, diag_var, other_var):
        if not hasattr(mat_a_var, 'eig_data'):
            evals, evec = mat_a_var.data.eig(eigenvectors=True)
            mat_a_var.eig_data = torch.cat([evec, evals[:, 0]], 1)
        if not hasattr(mat_b_var, 'eig_data'):
            evals, evec = mat_b_var.data.eig(eigenvectors=True)
            mat_b_var.eig_data = torch.cat([evec, evals[:, 0]], 1)

        # Switch the variable data with eig data, for computation
        # Save original data in function
        self.mat_a = mat_a_var.data
        self.mat_b = mat_b_var.data
        mat_a_var.data = mat_a_var.eig_data
        mat_b_var.data = mat_b_var.eig_data
        res = super(KronInvmm, self).__call__(mat_a_var, mat_b_var, diag_var, other_var)

        # Revert back to original data
        mat_a_var.data = self.mat_a
        mat_b_var.data = self.mat_b
        return res


Invmm.register_lazy_function((KronVariable, Variable), KronInvmm)
