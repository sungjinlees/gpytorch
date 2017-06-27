import math
import torch
from torch.autograd import Variable, Function
from gpytorch.lazy import KronVariable
from gpytorch.utils.kron import kron_forward, kron_backward
from .invmv import Invmv
from .invmm import KronInvmm


class ExactGPMarginalLogLikelihood(Invmv):
    def forward(self, chol_mat, y):
        mat_inv_y = y.potrs(chol_mat)
        res = mat_inv_y.dot(y) # Inverse quad
        res += chol_mat.diag().log_().sum() * 2 # Log determinant
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.save_for_backward(chol_mat, y)
        self.mat_inv_y = mat_inv_y
        return chol_mat.new().resize_(1).fill_(res)


    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        chol_matrix, y = self.saved_tensors
        mat_inv_y = self.mat_inv_y

        mat_grad = None
        y_grad = None

        if self.needs_input_grad[0]:
            mat_grad = torch.ger(y.view(-1), mat_inv_y.view(-1))
            mat_grad.add_(-torch.eye(*mat_grad.size()))
            mat_grad = mat_grad.potrs(chol_matrix, out=mat_grad)
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            y_grad = mat_inv_y.mul_(-grad_output_value)

        return mat_grad, y_grad


class KronExactGPMarginalLogLikelihood(KronInvmm):
    def forward(self, mat_a_eig_data, mat_b_eig_data, diag, y):
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

        # Inverse quad
        evec_eval_inv_prod = evec / eval_with_noise.unsqueeze(0).expand(m * p, m * p)
        mat_inv_y = evec_eval_inv_prod.mv(evec.t().mv(y.view(-1)))
        res = mat_inv_y.dot(y.view(-1))

        # Log determinant
        res += eval_with_noise.log().sum()

        # Add constant and stuff
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.evec = evec
        self.evec_eval_inv_prod = evec_eval_inv_prod
        self.mat_inv_y = mat_inv_y
        self.save_for_backward(y)
        return y.new().resize_(1).fill_(res)


    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        mat_a_grad = None
        mat_b_grad = None
        diag_grad = None
        y_grad = None

        evec = self.evec
        evec_eval_inv_prod = self.evec_eval_inv_prod
        mat_inv_y = self.mat_inv_y
        y, = self.saved_tensors

        if any(self.needs_input_grad[:3]):
            kron_grad = torch.ger(y.view(-1), mat_inv_y)
            kron_grad.add_(-torch.eye(*kron_grad.size()))
            kron_grad = torch.mm(evec_eval_inv_prod, evec.t().mm(kron_grad))
            kron_grad.mul_(0.5 * grad_output_value)

            mat_a_grad, mat_b_grad, diag_grad = kron_backward(
                    self.mat_a,
                    self.mat_b,
                    kron_grad,
                    self.needs_input_grad[:3]
            )
        
        if self.needs_input_grad[3]:
            y_grad = mat_inv_y.mul_(-grad_output_value)

        return mat_a_grad, mat_b_grad, diag_grad, y_grad


ExactGPMarginalLogLikelihood.register_lazy_function((KronVariable, Variable),
        KronExactGPMarginalLogLikelihood)
